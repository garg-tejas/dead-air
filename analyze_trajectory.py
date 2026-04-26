import json

def parse_trajectory(path):
    """Parse pretty-printed JSON objects from trajectory file."""
    records = []
    buffer = []
    with open(path, 'r') as f:
        for line in f:
            buffer.append(line)
            if line.strip() == '}':
                try:
                    records.append(json.loads(''.join(buffer)))
                    buffer = []
                except json.JSONDecodeError:
                    pass
    return records

if __name__ == '__main__':
    records = parse_trajectory('outputs/unsloth_test/trajectory.jsonl')
    
    # Find steps with active calls and see what the model outputs
    steps_with_calls = []
    steps_without_calls = []
    
    for r in records:
        prompt = r['prompt']
        has_calls = '## Active Calls' in prompt and prompt.split('## Active Calls')[1].strip().startswith('- Call')
        
        comp = r['completion']
        lines = comp.strip().split('\n')
        last_line = lines[-1].strip() if lines else ""
        
        info = {
            'batch': r['batch'],
            'episode': r['episode'],
            'step': r['step'],
            'last_line': last_line,
            'comp_len': len(comp),
            'starts_with_think': comp.strip().startswith('<think>'),
        }
        
        if has_calls:
            steps_with_calls.append(info)
        else:
            steps_without_calls.append(info)
    
    print(f"Steps WITH calls: {len(steps_with_calls)}")
    print(f"Steps WITHOUT calls: {len(steps_without_calls)}")
    
    print("\n=== Samples WITH calls ===")
    for s in steps_with_calls[:10]:
        print(f"  B{s['batch']}E{s['episode']}S{s['step']}: len={s['comp_len']}, last={s['last_line'][:80]}")
    
    print("\n=== Samples WITHOUT calls ===")
    for s in steps_without_calls[:10]:
        print(f"  B{s['batch']}E{s['episode']}S{s['step']}: len={s['comp_len']}, last={s['last_line'][:80]}")
    
    # Check if last line is valid JSON
    def is_valid_json_action(line):
        try:
            obj = json.loads(line)
            return 'action_type' in obj
        except:
            return False
    
    valid_with = sum(1 for s in steps_with_calls if is_valid_json_action(s['last_line']))
    valid_without = sum(1 for s in steps_without_calls if is_valid_json_action(s['last_line']))
    
    print(f"\nValid JSON rate WITH calls: {valid_with}/{len(steps_with_calls)} ({100*valid_with/max(len(steps_with_calls),1):.1f}%)")
    print(f"Valid JSON rate WITHOUT calls: {valid_without}/{len(steps_without_calls)} ({100*valid_without/max(len(steps_without_calls),1):.1f}%)")
    
    # Check action types for valid ones
    actions = {}
    for s in steps_with_calls + steps_without_calls:
        if is_valid_json_action(s['last_line']):
            try:
                act = json.loads(s['last_line'])['action_type']
                actions[act] = actions.get(act, 0) + 1
            except:
                pass
    print(f"\nAction distribution: {actions}")
    
    # Check for prompt echoing
    echo_count = 0
    for r in records:
        if 'You are an emergency dispatch commander' in r['completion']:
            echo_count += 1
    print(f"\nPrompt echoing occurrences: {echo_count}")
