import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import json
import numpy as np
from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy, RoundRobinSelectPolicy
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, NoMutatePolicy, MutateWeightedSamplingPolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
from gptfuzzer.fuzzer import GPTFuzzer
from gptfuzzer.utils.predict import MatchPredictor, AccessGrantedPredictor
from gptfuzzer.llm import OpenAILLM, OpenAIEmbeddingLLM
from PromptFuzz.utils import constants

import random
random.seed(100)
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)


def run_fuzzer(args):
    random_seed = int(getattr(args, 'random_seed', 100))
    random.seed(random_seed)
    np.random.seed(random_seed)
        
    api_base = getattr(args, 'api_base', None) or getattr(constants, 'openai_base_url', None)
    if api_base is None:
        api_base = getattr(constants, 'openai_base_url', None)

    mutate_model = OpenAILLM(args.model_path, args.openai_key, api_base=api_base)
    target_model = OpenAILLM(args.model_path, args.openai_key, api_base=api_base)

    if args.mode == 'hijacking':
        predictor = AccessGrantedPredictor()
    elif args.mode == 'extraction':
        predictor = MatchPredictor()
    else:
        raise ValueError("Invalid mode")
    
    if getattr(args, 'result_file', None):
        save_path = args.result_file
    elif args.baseline != None:
        save_path = f'./Results/{args.phase}/{args.mode}/baseline/{args.baseline}/{args.index}.csv' if not args.all_defenses else f'./Results/{args.phase}/{args.mode}/baseline/{args.baseline}/all_results.csv'    
    else:
        save_path = f'./Results/{args.phase}/{args.mode}/{args.index}.csv' if not args.all_defenses else f'./Results/{args.phase}/{args.mode}/all_results.csv'    
    
    print("The save path is: ", save_path)

    # check if the directory exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # Explicit paths support leakage-free paper splits; phase paths are kept
    # for reproducing the original PromptFuzz protocol.
    if getattr(args, 'defense_file', None):
        defense = args.defense_file
    elif args.phase == 'init':
        if args.baseline == 'human_expert' or args.baseline == 'gcg' or args.baseline == 'initial_seed':
            defense = f'./Datasets/{args.mode}_focus_defense.jsonl'    
        else:
            defense = f'./Datasets/{args.mode}_robustness_dataset.jsonl'
    elif args.phase == 'preparation':
        defense = f'./Datasets/{args.mode}_preparation_defense.jsonl'
    elif args.phase == 'focus':
        defense = f'./Datasets/{args.mode}_focus_defense.jsonl'
    else:
        raise ValueError("Invalid phase")
    
    with open(defense, 'r', encoding='utf-8') as f:
        defenses = [json.loads(line) for line in f.readlines()]
    if args.all_defenses:
        max_defenses = int(getattr(args, 'max_defenses', 0))
        args.defenses = defenses[:max_defenses] if max_defenses > 0 else defenses
    else:
        defenses = defenses[args.index]
        args.defenses = [defenses]
    defense_ids = [item.get('sample_id', position)
                   for position, item in enumerate(args.defenses)]
    print(f"Defense count={len(args.defenses)} sample_ids={defense_ids[:10]}")
    if args.no_mutate:
        assert args.phase == 'init'

    # load the initial seed
    if getattr(args, 'seed_file', None):
        initial_seed_path = args.seed_file
    elif args.phase == 'init':
        if args.baseline == 'human_expert' or args.baseline == 'gcg':
            initial_seed_path = f'./Datasets/{args.mode}_{args.baseline}_baseline.jsonl'
        elif args.baseline == 'initial_seed':
            initial_seed_path = f'./Datasets/{args.mode}_preparation_seed.jsonl'
        else:
            initial_seed_path = f'./Datasets/{args.mode}_robustness_dataset.jsonl'
    elif args.phase == 'focus':
        initial_seed_path = f'./Datasets/{args.mode}_focus_seed.jsonl'
    elif args.phase == 'preparation':
        initial_seed_path = f'./Datasets/{args.mode}_preparation_seed.jsonl'
        
    with open(initial_seed_path, 'r', encoding='utf-8') as f:
        initial_seed = [json.loads(line)['attack'] for line in f.readlines()]
    
    mutator_list = [
            OpenAIMutatorCrossOver(mutate_model), 
            OpenAIMutatorExpand(mutate_model),
            OpenAIMutatorGenerateSimilar(mutate_model),
            OpenAIMutatorRephrase(mutate_model),
            OpenAIMutatorShorten(mutate_model)
            ]
    
    mutate_policy = MutateRandomSinglePolicy(
            mutator_list,
            concatentate=args.concatenate,
        )
    select_policy = MCTSExploreSelectPolicy()
    
    if args.no_mutate:
        mutate_policy = NoMutatePolicy()
        args.energy = 1
        if not getattr(args, 'strict_budget', False):
            args.max_query = len(initial_seed)
        select_policy = RoundRobinSelectPolicy()
        
    if args.phase == 'preparation':
        args.energy = 1
        if not getattr(args, 'strict_budget', False):
            args.max_query = len(initial_seed) * len(args.defenses) * 10
        select_policy = RoundRobinSelectPolicy()
        
    if args.phase == 'focus':
        args.energy = 5
        if not getattr(args, 'strict_budget', False):
            args.max_query = len(args.defenses) * 1000
        select_policy = MCTSExploreSelectPolicy()
        
        few_shot_examples = pd.read_csv(f'./Datasets/{args.mode}_few_shot_example.csv')
        embedding_model = OpenAIEmbeddingLLM("text-embedding-ada-002", args.openai_key)
        mutate_policy = MutateWeightedSamplingPolicy(
            mutator_list,
            weights=args.mutator_weights,
            few_shot=args.few_shot,
            few_shot_num=args.few_shot_num,
            few_shot_file=few_shot_examples,
            concatentate=args.concatenate,
            retrieval_method=args.retrieval_method,
            cluster_num=args.cluster_num,
            embedding_model=embedding_model,
        )
        
    update_pool = True if args.phase == 'focus' else False
    
    fuzzer = GPTFuzzer(
        defenses=args.defenses,
        target=target_model,
        predictor=predictor,
        initial_seed=initial_seed,
        result_file=save_path,
        mutate_policy=mutate_policy,
        select_policy=select_policy,
        energy=args.energy,
        max_jailbreak=args.max_jailbreak,
        max_query=args.max_query,
        update_pool=update_pool,
        dynamic_allocate=args.dynamic_allocate,
        threshold_coefficient=args.threshold_coefficient,
        target_max_tokens=args.target_max_tokens,
    )

    summary = fuzzer.run()
    summary['target_model_requests'] = target_model.request_count
    summary['mutation_model_requests'] = mutate_model.request_count
    summary['transport_requests'] = target_model.request_count + mutate_model.request_count
    summary['target_model_generations'] = target_model.generation_count
    summary['mutation_model_generations'] = mutate_model.generation_count
    # One transport request can generate multiple completions. Counting actual
    # generations gives a provider-independent compute proxy.
    summary['total_llm_calls'] = target_model.generation_count + mutate_model.generation_count
    summary['target_prompt_tokens'] = target_model.prompt_tokens
    summary['target_completion_tokens'] = target_model.completion_tokens
    summary['mutation_prompt_tokens'] = mutate_model.prompt_tokens
    summary['mutation_completion_tokens'] = mutate_model.completion_tokens
    summary['prompt_tokens'] = target_model.prompt_tokens + mutate_model.prompt_tokens
    summary['completion_tokens'] = target_model.completion_tokens + mutate_model.completion_tokens
    summary['total_tokens'] = target_model.total_tokens + mutate_model.total_tokens
    summary['result_file'] = save_path
    summary['random_seed'] = random_seed
    print('PROMPTFUZZ_SUMMARY=' + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary
