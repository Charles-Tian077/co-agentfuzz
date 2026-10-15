import sys
import os

# A generated prompt/response may contain characters that the Windows console
# code page cannot represent.  Keep diagnostic output from terminating a run;
# result files retain the complete text in UTF-8.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(errors='backslashreplace')

# Add the repo root to sys.path so the project can be launched from any working directory.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import argparse
from PromptFuzz.Fuzzer.promptfuzz import run_fuzzer
from PromptFuzz.utils import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--index', type=int, default=0, help='The index of the target prompt')
    parser.add_argument('--phase', choices=['preparation', 'focus', 'init'], default='init', help='The phase of the fuzzing process')
    parser.add_argument('--mode', choices=['hijacking', 'extraction'], default='extraction', help='The mode of the fuzzing process')
    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI API Key or dummy key for local OpenAI-compatible servers')
    parser.add_argument('--api_base', '--base_url', dest='api_base', type=str, default=None, help='OpenAI-compatible base URL for local/free models, e.g. http://localhost:11434/v1')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo-0125', help='target model path')
    parser.add_argument('--max_query', type=int, default=1000,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=9999999, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    parser.add_argument("--no_mutate", action='store_true', help='Do not mutate the prompt')
    parser.add_argument("--all_defenses", action='store_true', help='Run all defenses')
    parser.add_argument("--concatenate", action='store_true', help='Concatenate the prompt')
    parser.add_argument("--few_shot", action='store_true', help='Use few shot learning')
    parser.add_argument("--retrieval_method", choices=['random', 'cosine_similarity', 'cluster'], default='random', help='The retrieval method')
    parser.add_argument("--cluster_num", type=int, default=5, help='The number of clusters for retrieval')
    parser.add_argument("--few_shot_num", type=int, default=3)
    parser.add_argument("--dynamic_allocate", action='store_true', help='Dynamic allocate the energy')
    parser.add_argument("--threshold_coefficient", type=float, default=0.5, help='The threshold coefficient')
    parser.add_argument("--mutator_weights", type=float, nargs='+', default=[0.2, 0.2, 0.2, 0.2, 0.2], help='The weights for the mutator selection')
    parser.add_argument("--baseline", type=str, default=None, help='The baseline model')
    parser.add_argument("--strict_budget", action="store_true",
                        help="Respect --max_query instead of applying phase defaults")
    parser.add_argument("--result_file", type=str, default=None,
                        help="Optional result CSV path for controlled comparisons")
    parser.add_argument("--seed_file", type=str, default=None,
                        help="Explicit JSONL attack-seed path")
    parser.add_argument("--defense_file", type=str, default=None,
                        help="Explicit JSONL defense path")
    parser.add_argument("--max_defenses", type=int, default=0,
                        help="Limit --all_defenses to the first N records; 0 means all")
    parser.add_argument("--random_seed", type=int, default=100,
                        help="Random seed for controlled comparisons")
    parser.add_argument("--target_max_tokens", type=int, default=512,
                        help="Maximum target-response tokens (set equally across methods)")

    args = parser.parse_args()
    
    if args.openai_key is None:
        args.openai_key = constants.openai_key
        
    run_fuzzer(args)
