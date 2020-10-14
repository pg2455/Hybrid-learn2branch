import os
import sys
import importlib
import argparse
import csv
import math
import numpy as np
import time
import pickle
import pathlib
import pyscipopt as scip

import tensorflow as tf
import torch
import utilities
from utilities import _get_model_type


class PolicyBranching(scip.Branchrule):

    def __init__(self, policy, device):
        super().__init__()

        self.policy_type = policy['type']
        self.policy_name = policy['name']
        self.device = device

        model = policy['model']
        model.restore_state(policy['parameters'])
        model.to(device)
        model.eval()
        self.policy_get_root_params = model.get_params
        self.policy_predict = model.predict

        self.teacher = None
        if policy['teacher_type'] is not None:
            self.teacher = policy['teacher_model']
            self.teacher.restore_state(policy['teacher_parameters'])
            self.teacher.to(device)
            self.teacher.eval()

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):

        candidate_vars, *_ = self.model.getPseudoBranchCands()
        candidate_mask = [var.getCol().getIndex() for var in candidate_vars]
        state = utilities.extract_state(self.model, self.state_buffer)
        c,e,v =  state
        if self.model.getNNodes() == 1:
            state = (
                torch.as_tensor(c['values'], dtype=torch.float32),
                torch.as_tensor(e['indices'], dtype=torch.long),
                torch.as_tensor(e['values'], dtype=torch.float32),
                torch.as_tensor(v['values'], dtype=torch.float32),
                torch.as_tensor(c['values'].shape[0], dtype=torch.int32),
                torch.as_tensor(v['values'].shape[0], dtype=torch.int32),
            )
            state = map(lambda x:x.to(self.device), state)
            with torch.no_grad():
                if self.teacher is not None:
                    root_feats, _ = self.teacher(state)
                    state = root_feats
                self.root_params = self.policy_get_root_params(state)

        v = v['values'][candidate_mask]
        state_khalil = utilities.extract_khalil_variable_features(self.model, candidate_vars, self.khalil_root_buffer)

        var_feats = np.concatenate([v, state_khalil, np.ones((v.shape[0],1))], axis=1)
        var_feats = utilities._preprocess(var_feats, mode="min-max-2")
        var_feats = torch.as_tensor(var_feats, dtype=torch.float32).to(device)

        with torch.no_grad():
            var_logits = self.policy_predict(var_feats, self.root_params[candidate_mask]).cpu().numpy()

        best_var = candidate_vars[var_logits.argmax()]
        self.model.branchVar(best_var)
        result = scip.SCIP_RESULT.BRANCHED

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '-s', '--seed',
        help='seed for parallelizing the evaluation. Uses all seeds if not provided.',
        type=int,
        default=-1
    )
    parser.add_argument(
        '-l', '--level',
        help='size of instances to evaluate. Default is all.',
        type=str,
        default='all',
        choices=['all', 'small', 'medium', 'big']
    )
    parser.add_argument(
        '--trained_models',
        help='Directory of trained models.',
        type=str,
        default='trained_models/'
    )
    parser.add_argument(
        '-m', '--model_string',
        help='searches for this string in respective trained_models folder',
        type=str,
        default='',
    )
    parser.add_argument(
        '--model_name',
        help='searches for this model_name in respective trained_models folder',
        type=str,
        default='',
    )
    args = parser.parse_args()

    teacher_model = "baseline_torch" # used if pretrained model is used
    instances = []
    seeds = [0, 1, 2]
    time_limit = 2700

    result_dir = f"eval_results/{args.problem}"
    os.makedirs(result_dir, exist_ok=True)
    device = "CPU" if args.gpu == -1 else "GPU"
    result_file = f"{result_dir}/hybrid_{device}_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    ### MODELS TO EVALUATE ###
    basedir = f"{args.trained_models}/{args.problem}"
    if args.model_string != "":
        models_to_evaluate = [y for y in pathlib.Path(basedir).iterdir() if args.model_string in y.name]
        assert len(models_to_evaluate) > 0, f"no model matched the model_string: {args.model_string} in {basedir}"
    elif args.model_name != "":
        model_path = pathlib.Path(f"{basedir}/{args.model_name}")
        assert model_path.exists(), f"path: {model_path} doesn't exist in {basedir}"
        models_to_evaluate = [model_path]
    else:
        models_to_evaluate = [y for y in pathlib.Path(f"{basedir}").iterdir()]
        assert len(models_to_evaluate) > 0, f"no model found in {basedir}"

    if args.problem == 'setcover':
        instances += [{'type': 'small', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]

    elif args.problem == 'cauctions':
        instances += [{'type': 'small', 'path': f"data/instances/cauctions/transfer_100_500/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/cauctions/transfer_200_1000/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/cauctions/transfer_300_1500/instance_{i+1}.lp"} for i in range(20)]

    elif args.problem == 'facilities':
        instances += [{'type': 'small', 'path': f"data/instances/facilities/transfer_100_100_5/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/facilities/transfer_200_100_5/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/facilities/transfer_400_100_5/instance_{i+1}.lp"} for i in range(20)]

    elif args.problem == 'indset':
        instances += [{'type': 'small', 'path': f"data/instances/indset/transfer_750_4/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/indset/transfer_1000_4/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/indset/transfer_1500_4/instance_{i+1}.lp"} for i in range(20)]

    else:
        raise NotImplementedError

    ### SEEDS TO EVALUATE ###
    if args.seed != -1:
        seeds = [args.seed]

    ### PROBLEM SIZES TO EVALUATE ###
    if args.level != "all":
        instances = [x for x in instances if x['type'] == args.level]

    branching_policies = []

    ### MODEL PARAMETERS
    for model_path in models_to_evaluate:
        try:
            model_type = _get_model_type(model_path.name)
        except ValueError as e:
            print(e, "skipping ...")
            continue

        for seed in seeds:
            policy = {
                'type': model_type,
                'name': model_path.name,
                'seed': seed,
                'parameters': str(model_path / f"{seed}/best_params.pkl"),
                'teacher_type': None,
                'teacher_parameters': None
            }
            if "-pre" in model_type:
                policy['teacher_type'] = teacher_model
                policy['teacher_parameters'] = f"{basedir}/{teacher_model}/{seed}/best_params.pkl"
            branching_policies.append(policy)

    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")

    ### NUMPY / TORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    for policy in branching_policies:
        if policy['type'] not in loaded_models:
            sys.path.insert(0, os.path.abspath(f"models/{policy['type']}"))
            import model
            importlib.reload(model)
            loaded_models[policy['type']] = model.Policy()
            del sys.path[0]
            loaded_models[policy['type']].to(device)
            loaded_models[policy['type']].eval()

        if (
            policy['teacher_type'] is not None
            and policy['teacher_type'] not in loaded_models
        ):
            sys.path.insert(0, os.path.abspath(f"models/{policy['teacher_type']}"))
            import model
            importlib.reload(model)
            loaded_models[policy['teacher_type']] = model.GCNPolicy()
            del sys.path[0]
            loaded_models[policy['teacher_type']].to(device)
            loaded_models[policy['teacher_type']].eval()

        policy['model'] = loaded_models[policy['type']]
        policy['teacher_model'] = loaded_models[policy['teacher_type']] if policy['teacher_type'] is not None else None

    print("running SCIP...")

    fieldnames = [
        'problem',
        'device',
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'stime',
        'gap',
        'status',
        'ndomchgs',
        'ncutoffs',
        'walltime',
        'proctime',
    ]

    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            for policy in branching_policies:
                torch.manual_seed(policy['seed'])

                m = scip.Model()
                m.setIntParam('display/verblevel', 0)
                m.readProblem(f"{instance['path']}")
                utilities.init_scip_params(m, seed=policy['seed'])
                m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
                m.setRealParam('limits/time', time_limit)

                brancher = PolicyBranching(policy, device)
                m.includeBranchrule(
                    branchrule=brancher,
                    name=f"{policy['type']}:{policy['name']}",
                    desc=f"Custom MLPOpt branching policy.",
                    priority=666666, maxdepth=-1, maxbounddist=1)

                walltime = time.perf_counter()
                proctime = time.process_time()

                m.optimize()

                walltime = time.perf_counter() - walltime
                proctime = time.process_time() - proctime

                stime = m.getSolvingTime()
                nnodes = m.getNNodes()
                nlps = m.getNLPs()
                gap = m.getGap()
                status = m.getStatus()
                ndomchgs = brancher.ndomchgs
                ncutoffs = brancher.ncutoffs

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'type': instance['type'],
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'stime': stime,
                    'gap': gap,
                    'status': status,
                    'ndomchgs': ndomchgs,
                    'ncutoffs': ncutoffs,
                    'walltime': walltime,
                    'proctime': proctime,
                    'problem':args.problem,
                    'device': device
                })

                csvfile.flush()
                m.freeProb()

                print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} ({nnodes+2*(ndomchgs+ncutoffs)}) nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")
