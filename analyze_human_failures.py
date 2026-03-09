"""
分析human level问题中agent回答失败的情况，找出缺点和失败模式。
"""
import os
import sys
import json
import csv
import argparse
from collections import defaultdict, Counter

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def load_csv(path):
    """Load CSV, return list of dicts."""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def analyze_method(method_name, results_dir, eval_dir, query_data):
    print(f"\n{'='*80}")
    print(f"  分析 Agent: {method_name}")
    print(f"{'='*80}")

    # Load eval CSVs
    schema_csv = load_csv(os.path.join(eval_dir, 'schema.csv'))
    commonsense_csv = load_csv(os.path.join(eval_dir, 'commonsense.csv'))
    logical_csv = load_csv(os.path.join(eval_dir, 'logical_py.csv'))

    # Build lookup dicts
    schema_dict = {r['data_id']: int(r['schema']) for r in schema_csv}
    
    # Commonsense: check if all fields are 0 (pass)
    commonsense_fields = [k for k in commonsense_csv[0].keys() if k != 'data_id']
    commonsense_dict = {}
    commonsense_detail = {}
    for r in commonsense_csv:
        all_pass = all(int(r[f]) == 0 for f in commonsense_fields)
        commonsense_dict[r['data_id']] = 1 if all_pass else 0
        failed_fields = [f for f in commonsense_fields if int(r[f]) != 0]
        commonsense_detail[r['data_id']] = failed_fields

    # Logical: check if all logic constraints pass
    logical_fields = [k for k in logical_csv[0].keys() if k != 'data_id']
    logical_dict = {}
    logical_detail = {}
    for r in logical_csv:
        vals = [r[f] for f in logical_fields if r[f] != '']
        all_pass = all(int(v) == 1 for v in vals) if vals else False
        logical_dict[r['data_id']] = 1 if all_pass else 0
        failed_indices = [i for i, v in enumerate(vals) if int(v) == 0]
        logical_detail[r['data_id']] = (len(vals), failed_indices)

    all_ids = list(schema_dict.keys())
    total = len(all_ids)

    # ===== 1. 总体统计 =====
    schema_pass = sum(schema_dict.values())
    comm_pass = sum(commonsense_dict.values())
    logi_pass = sum(logical_dict.values())
    all_pass_ids = [qid for qid in all_ids 
                    if schema_dict.get(qid, 0) == 1 
                    and commonsense_dict.get(qid, 0) == 1 
                    and logical_dict.get(qid, 0) == 1]
    
    print(f"\n--- 总体通过率 ---")
    print(f"  总问题数: {total}")
    print(f"  Schema通过: {schema_pass}/{total} ({100*schema_pass/total:.1f}%)")
    print(f"  常识约束通过: {comm_pass}/{total} ({100*comm_pass/total:.1f}%)")
    print(f"  逻辑约束通过: {logi_pass}/{total} ({100*logi_pass/total:.1f}%)")
    print(f"  全部通过: {len(all_pass_ids)}/{total} ({100*len(all_pass_ids)/total:.1f}%)")

    # ===== 2. 分析结果文件（超时vs格式错误vs有内容） =====
    timeout_count = 0
    format_error_count = 0
    has_plan_count = 0
    empty_result_count = 0
    
    result_details = {}
    for qid in all_ids:
        result_file = os.path.join(results_dir, f"{qid}.json")
        result = load_json(result_file)
        if result is None:
            empty_result_count += 1
            result_details[qid] = 'missing'
            continue
        
        if result.get('time_out_flag', False):
            timeout_count += 1
            result_details[qid] = 'timeout'
        elif 'itinerary' not in result:
            format_error_count += 1
            result_details[qid] = 'no_plan'
        else:
            has_plan_count += 1
            result_details[qid] = 'has_plan'
    
    print(f"\n--- 结果文件分析 ---")
    print(f"  超时 (timeout): {timeout_count}/{total} ({100*timeout_count/total:.1f}%)")
    print(f"  无计划 (no itinerary): {format_error_count}/{total} ({100*format_error_count/total:.1f}%)")
    print(f"  有计划: {has_plan_count}/{total} ({100*has_plan_count/total:.1f}%)")
    print(f"  结果缺失: {empty_result_count}/{total}")

    # ===== 2b. 非超时查询的通过率 =====
    non_timeout_ids = [qid for qid in all_ids if result_details.get(qid) != 'timeout' and result_details.get(qid) != 'missing']
    nt = len(non_timeout_ids)
    if nt > 0:
        nt_schema_pass  = sum(schema_dict.get(q, 0) for q in non_timeout_ids)
        nt_comm_pass    = sum(commonsense_dict.get(q, 0) for q in non_timeout_ids)
        nt_logi_pass    = sum(logical_dict.get(q, 0) for q in non_timeout_ids)
        nt_all_pass     = sum(
            1 for q in non_timeout_ids
            if schema_dict.get(q, 0) == 1
            and commonsense_dict.get(q, 0) == 1
            and logical_dict.get(q, 0) == 1
        )
        print(f"\n--- 非超时查询通过率 (共 {nt} 个有输出的查询) ---")
        print(f"  Schema通过: {nt_schema_pass}/{nt} ({100*nt_schema_pass/nt:.1f}%)")
        print(f"  常识约束通过: {nt_comm_pass}/{nt} ({100*nt_comm_pass/nt:.1f}%)")
        print(f"  逻辑约束通过: {nt_logi_pass}/{nt} ({100*nt_logi_pass/nt:.1f}%)")
        print(f"  全部通过: {nt_all_pass}/{nt} ({100*nt_all_pass/nt:.1f}%)")

    # ===== 2c. 高约束查询 (6-9个约束) 的专项分析 =====
    high_constraint_ids = [qid for qid in all_ids
                           if logical_detail.get(qid, (0, []))[0] >= 6]
    hc = len(high_constraint_ids)
    if hc > 0:
        hc_schema   = sum(schema_dict.get(q, 0) for q in high_constraint_ids)
        hc_comm     = sum(commonsense_dict.get(q, 0) for q in high_constraint_ids)
        hc_logi     = sum(logical_dict.get(q, 0) for q in high_constraint_ids)
        hc_all      = sum(1 for q in high_constraint_ids
                          if schema_dict.get(q,0)==1
                          and commonsense_dict.get(q,0)==1
                          and logical_dict.get(q,0)==1)
        hc_timeout  = sum(1 for q in high_constraint_ids if result_details.get(q) == 'timeout')
        hc_no_plan  = sum(1 for q in high_constraint_ids if result_details.get(q) == 'no_plan')
        hc_has_plan = sum(1 for q in high_constraint_ids if result_details.get(q) == 'has_plan')
        hc_missing  = sum(1 for q in high_constraint_ids if result_details.get(q) == 'missing')
        print(f"\n--- 高约束查询 (≥6个约束, 共 {hc} 个) 专项分析 ---")
        print(f"  [通过率]")
        print(f"    Schema通过: {hc_schema}/{hc} ({100*hc_schema/hc:.1f}%)")
        print(f"    常识约束通过: {hc_comm}/{hc} ({100*hc_comm/hc:.1f}%)")
        print(f"    逻辑约束通过: {hc_logi}/{hc} ({100*hc_logi/hc:.1f}%)")
        print(f"    全部通过: {hc_all}/{hc} ({100*hc_all/hc:.1f}%)")
        print(f"  [结果文件分布]")
        print(f"    超时 (timeout): {hc_timeout}/{hc} ({100*hc_timeout/hc:.1f}%)")
        print(f"    无计划 (no itinerary): {hc_no_plan}/{hc} ({100*hc_no_plan/hc:.1f}%)")
        print(f"    有计划: {hc_has_plan}/{hc} ({100*hc_has_plan/hc:.1f}%)")
        print(f"    结果缺失: {hc_missing}/{hc}")
        # per-n breakdown
        print(f"  [按约束数量细分]")
        for n in sorted(set(logical_detail.get(q,(0,[]))[0] for q in high_constraint_ids)):
            sub = [q for q in high_constraint_ids if logical_detail.get(q,(0,[]))[0] == n]
            sub_pass = sum(1 for q in sub if logical_dict.get(q,0)==1)
            sub_tout = sum(1 for q in sub if result_details.get(q)=='timeout')
            print(f"    {n}个约束: {len(sub)}题, 逻辑通过={sub_pass}, 超时={sub_tout}")

    # ===== 3. 失败问题的模式分析 =====
    # 分类失败类型
    only_schema_fail = []
    only_comm_fail = []
    only_logi_fail = []
    schema_comm_fail = []
    schema_logi_fail = []
    comm_logi_fail = []
    all_fail = []
    
    for qid in all_ids:
        s = schema_dict.get(qid, 0)
        c = commonsense_dict.get(qid, 0)
        l = logical_dict.get(qid, 0)
        
        if s == 1 and c == 1 and l == 1:
            continue  # all pass
        
        if s == 0 and c == 1 and l == 1:
            only_schema_fail.append(qid)
        elif s == 1 and c == 0 and l == 1:
            only_comm_fail.append(qid)
        elif s == 1 and c == 1 and l == 0:
            only_logi_fail.append(qid)
        elif s == 0 and c == 0 and l == 1:
            schema_comm_fail.append(qid)
        elif s == 0 and c == 1 and l == 0:
            schema_logi_fail.append(qid)
        elif s == 1 and c == 0 and l == 0:
            comm_logi_fail.append(qid)
        elif s == 0 and c == 0 and l == 0:
            all_fail.append(qid)
    
    print(f"\n--- 失败模式分布 ---")
    print(f"  仅Schema失败: {len(only_schema_fail)}")
    print(f"  仅常识失败: {len(only_comm_fail)}")
    print(f"  仅逻辑失败: {len(only_logi_fail)}")
    print(f"  Schema+常识失败: {len(schema_comm_fail)}")
    print(f"  Schema+逻辑失败: {len(schema_logi_fail)}")
    print(f"  常识+逻辑失败: {len(comm_logi_fail)}")
    print(f"  全部失败: {len(all_fail)}")

    # ===== 4. 常识约束错误的具体分类 =====
    comm_error_counter = Counter()
    for qid in all_ids:
        if commonsense_dict.get(qid, 0) == 0:
            for field in commonsense_detail.get(qid, []):
                comm_error_counter[field] += 1
    
    print(f"\n--- 常识约束失败类型统计 (仅失败的问题) ---")
    comm_fail_total = total - comm_pass
    for field, count in comm_error_counter.most_common():
        print(f"  {field}: {count}/{comm_fail_total} ({100*count/comm_fail_total:.1f}%)")

    # ===== 5. 逻辑约束分析 =====
    # 统计每个问题有多少个逻辑约束，失败的是哪些
    logi_constraint_counts = Counter()
    logi_fail_position = Counter()
    for qid in all_ids:
        n_constraints, failed_idx = logical_detail.get(qid, (0, []))
        logi_constraint_counts[n_constraints] += 1
        for idx in failed_idx:
            logi_fail_position[f"logic_{idx}"] += 1
    
    print(f"\n--- 逻辑约束详情 (高复杂度样例) ---")
    max_n = max(logi_constraint_counts.keys()) if logi_constraint_counts else 0
    found_max = False
    for qid in all_ids:
        n_constraints, _ = logical_detail.get(qid, (0, []))
        if n_constraints == max_n and max_n > 0:
            found_max = True
            q_data = query_data.get(qid, {})
            print(f"  查询 UID: {qid} (拥有最多的 {max_n} 个约束)")
            print(f"  目标城市: {q_data.get('target_city', 'N/A')}")
            logic_py = q_data.get('hard_logic_py', [])
            for i, lp in enumerate(logic_py):
                lp_clean = lp.replace('\n', ' ')
                print(f"    logic_{i}: {lp_clean}")
            break
    if not found_max:
        print("  未找到匹配的高约束查询资料。")

    print(f"\n--- 逻辑约束数量分布 ---")
    for n, count in sorted(logi_constraint_counts.items()):
        print(f"  {n}个约束: {count}个问题")
    
    print(f"\n--- 逻辑约束失败位置分布 ---")
    # Sort by logic index instead of count for readability
    sorted_pos = sorted(logi_fail_position.items(), key=lambda x: int(x[0].split('_')[1]))
    for pos, count in sorted_pos:
        print(f"  {pos}: {count}次失败")

    # ===== 6. 按约束数量分析通过率 =====
    print(f"\n--- 按逻辑约束数量的通过率 ---")
    for n_constraints in sorted(logi_constraint_counts.keys()):
        relevant_ids = [qid for qid in all_ids 
                        if logical_detail.get(qid, (0, []))[0] == n_constraints]
        pass_count = sum(1 for qid in relevant_ids if logical_dict.get(qid, 0) == 1)
        print(f"  {n_constraints}个约束: {pass_count}/{len(relevant_ids)} ({100*pass_count/len(relevant_ids):.1f}%)")

    # ===== 6b. 非超时情况下按约束数量的通过率 =====
    non_timeout_set = set(non_timeout_ids)
    print(f"\n--- 非超时: 按逻辑约束数量的通过率 ---")
    for n_constraints in sorted(logi_constraint_counts.keys()):
        relevant_ids = [qid for qid in non_timeout_set
                        if logical_detail.get(qid, (0, []))[0] == n_constraints]
        if not relevant_ids:
            continue
        pass_count = sum(1 for qid in relevant_ids if logical_dict.get(qid, 0) == 1)
        print(f"  {n_constraints}个约束: {pass_count}/{len(relevant_ids)} ({100*pass_count/len(relevant_ids):.1f}%)")

    # ===== 6c. 按天数分析通过率 =====
    days_group = defaultdict(list)
    for qid in all_ids:
        q_data = query_data.get(qid, {})
        d = q_data.get('days', 'N/A')
        days_group[d].append(qid)

    print(f"\n--- 按天数的通过率 ---")
    for d in sorted(days_group.keys(), key=lambda x: (isinstance(x, str), x)):
        ids = days_group[d]
        n = len(ids)
        s_pass = sum(schema_dict.get(q, 0) for q in ids)
        c_pass = sum(commonsense_dict.get(q, 0) for q in ids)
        l_pass = sum(logical_dict.get(q, 0) for q in ids)
        a_pass = sum(1 for q in ids
                     if schema_dict.get(q, 0) == 1
                     and commonsense_dict.get(q, 0) == 1
                     and logical_dict.get(q, 0) == 1)
        print(f"  {d}天 ({n}题): Schema={s_pass}/{n}({100*s_pass/n:.1f}%)  "
              f"常识={c_pass}/{n}({100*c_pass/n:.1f}%)  "
              f"逻辑={l_pass}/{n}({100*l_pass/n:.1f}%)  "
              f"全部通过={a_pass}/{n}({100*a_pass/n:.1f}%)")

    # ===== 6d. 非超时: 按天数的通过率 =====
    print(f"\n--- 非超时: 按天数的通过率 ---")
    for d in sorted(days_group.keys(), key=lambda x: (isinstance(x, str), x)):
        ids = [q for q in days_group[d] if q in non_timeout_set]
        n = len(ids)
        if n == 0:
            continue
        s_pass = sum(schema_dict.get(q, 0) for q in ids)
        c_pass = sum(commonsense_dict.get(q, 0) for q in ids)
        l_pass = sum(logical_dict.get(q, 0) for q in ids)
        a_pass = sum(1 for q in ids
                     if schema_dict.get(q, 0) == 1
                     and commonsense_dict.get(q, 0) == 1
                     and logical_dict.get(q, 0) == 1)
        print(f"  {d}天 ({n}题): Schema={s_pass}/{n}({100*s_pass/n:.1f}%)  "
              f"常识={c_pass}/{n}({100*c_pass/n:.1f}%)  "
              f"逻辑={l_pass}/{n}({100*l_pass/n:.1f}%)  "
              f"全部通过={a_pass}/{n}({100*a_pass/n:.1f}%)")

    # ===== 6e. 按天数的失败原因分布 =====
    print(f"\n--- 按天数的失败原因分布 ---")
    for d in sorted(days_group.keys(), key=lambda x: (isinstance(x, str), x)):
        ids = days_group[d]
        n = len(ids)
        d_timeout = sum(1 for q in ids if result_details.get(q) == 'timeout')
        d_no_plan = sum(1 for q in ids if result_details.get(q) == 'no_plan')
        d_has_plan = sum(1 for q in ids if result_details.get(q) == 'has_plan')
        d_missing = sum(1 for q in ids if result_details.get(q) == 'missing')
        d_all_pass = sum(1 for q in ids
                         if schema_dict.get(q, 0) == 1
                         and commonsense_dict.get(q, 0) == 1
                         and logical_dict.get(q, 0) == 1)
        d_has_plan_fail = sum(1 for q in ids
                              if result_details.get(q) == 'has_plan'
                              and not (schema_dict.get(q, 0) == 1
                                       and commonsense_dict.get(q, 0) == 1
                                       and logical_dict.get(q, 0) == 1))
        print(f"  {d}天 ({n}题): 超时={d_timeout}  无计划={d_no_plan}  "
              f"有计划但失败={d_has_plan_fail}  全部通过={d_all_pass}  缺失={d_missing}")

    # ===== 6f. 按天数 x 约束数量 交叉分析 =====
    print(f"\n--- 按天数 x 约束数量 交叉通过率 ---")
    day_cons_matrix = defaultdict(lambda: defaultdict(list))
    for qid in all_ids:
        q_data = query_data.get(qid, {})
        d = q_data.get('days', 'N/A')
        nc = logical_detail.get(qid, (0, []))[0]
        day_cons_matrix[d][nc].append(qid)
    for d in sorted(day_cons_matrix.keys(), key=lambda x: (isinstance(x, str), x)):
        row_parts = []
        for nc in sorted(day_cons_matrix[d].keys()):
            ids = day_cons_matrix[d][nc]
            n = len(ids)
            a_pass = sum(1 for q in ids
                         if schema_dict.get(q, 0) == 1
                         and commonsense_dict.get(q, 0) == 1
                         and logical_dict.get(q, 0) == 1)
            row_parts.append(f"{nc}约束:{a_pass}/{n}")
        print(f"  {d}天: {', '.join(row_parts)}")

    # ===== 7. 超时问题的分析 =====
    print(f"\n--- 超时问题分析 ---")
    timeout_ids = [qid for qid in all_ids if result_details.get(qid) == 'timeout']
    if timeout_ids:
        # Check how many have LLM format errors
        total_format_errors = 0
        total_search_nodes = 0
        total_backtrack = 0
        for qid in timeout_ids:
            result_file = os.path.join(results_dir, f"{qid}.json")
            result = load_json(result_file)
            if result:
                total_format_errors += result.get('llm_rec_format_error_count', 0)
                total_search_nodes += result.get('search_nodes', 0)
                total_backtrack += result.get('backtrack_count', 0)
        
        avg_format_err = total_format_errors / len(timeout_ids) if timeout_ids else 0
        avg_search = total_search_nodes / len(timeout_ids) if timeout_ids else 0
        avg_backtrack = total_backtrack / len(timeout_ids) if timeout_ids else 0
        print(f"  超时问题数: {len(timeout_ids)}")
        print(f"  平均格式错误次数: {avg_format_err:.1f}")
        print(f"  平均搜索节点数: {avg_search:.0f}")
        print(f"  平均回溯次数: {avg_backtrack:.0f}")

    # ===== 8. 有计划但仍失败的问题详细分析 =====
    print(f"\n--- 有计划但仍失败的问题 ---")
    has_plan_but_fail = [qid for qid in all_ids 
                         if result_details.get(qid) == 'has_plan' 
                         and qid not in all_pass_ids]
    has_plan_and_pass = [qid for qid in all_ids
                         if result_details.get(qid) == 'has_plan'
                         and qid in all_pass_ids]
    print(f"  有计划且通过: {len(has_plan_and_pass)}")
    print(f"  有计划但部分失败: {len(has_plan_but_fail)}")
    
    # Show some examples of failures with plans
    print(f"\n--- 部分失败样例分析 (有计划但逻辑约束未通过) ---")
    sample_count = 0
    for qid in has_plan_but_fail[:10]:
        if logical_dict.get(qid, 0) == 0 and commonsense_dict.get(qid, 0) == 1:
            sample_count += 1
            result_file = os.path.join(results_dir, f"{qid}.json")
            result = load_json(result_file)
            
            # Get query data if available 
            q_data = query_data.get(qid, {})
            hard_logic_nl = q_data.get('hard_logic_nl', [])
            hard_logic_py = q_data.get('hard_logic_py', [])
            
            n_constraints, failed_idx = logical_detail.get(qid, (0, []))
            
            print(f"\n  [{sample_count}] Query: {qid}")
            if q_data:
                print(f"    查询内容: {q_data.get('query', 'N/A')[:100]}")
                print(f"    出发城市: {q_data.get('start_city', 'N/A')}")
                print(f"    目标城市: {q_data.get('target_city', 'N/A')}")
                print(f"    天数: {q_data.get('days', 'N/A')}")
                print(f"    人数: {q_data.get('people_number', 'N/A')}")
            print(f"    总逻辑约束: {n_constraints}, 失败约束索引: {failed_idx}")
            
            if hard_logic_nl:
                for i, nl in enumerate(hard_logic_nl):
                    status = "✗ FAIL" if i in failed_idx else "✓ PASS"
                    print(f"      约束{i} [{status}]: {nl}")
            
            if sample_count >= 5:
                break

    # ===== 9. 跨方法对比：失败ID重叠分析 =====
    return {
        'all_ids': all_ids,
        'all_pass_ids': set(all_pass_ids),
        'schema_fail': set(qid for qid in all_ids if schema_dict.get(qid, 0) == 0),
        'comm_fail': set(qid for qid in all_ids if commonsense_dict.get(qid, 0) == 0),
        'logi_fail': set(qid for qid in all_ids if logical_dict.get(qid, 0) == 0),
        'timeout_ids': set(timeout_ids),
        'commonsense_detail': commonsense_detail,
        'logical_detail': logical_detail,
    }


def main():
    # Load query data
    try:
        from chinatravel.data.load_datasets import load_query
        parser = argparse.ArgumentParser()
        parser.add_argument("--splits", type=str, default="human")
        parser.add_argument("--oracle_translation", action='store_true', default=True)
        args = parser.parse_args([])
        args.splits = "human"
        args.oracle_translation = True
        query_id_list, query_data = load_query(args)
        print(f"已加载 {len(query_id_list)} 个human level查询")
    except Exception as e:
        print(f"无法加载查询数据: {e}")
        query_data = {}
    
    # Analyze each method
    methods_info = {}
    for method in  ['LLMNeSy_gpt-5.1-chat', 'LLMNeSy_Gemini-3-Flash-Preview','LLMNeSy_Qwen3-8B','LLMNeSy_deepseek']:
    # for method in ['LLMNeSy_Qwen3-8B', 'LLMNeSy_deepseek', 'LLMNeSy_Qwen3-8B_oracletranslation', 'LLMNeSy_deepseek_oracletranslation' ]:
        results_dir = f'results/{method}'
        eval_dir = f'eval_res/splits_human/{method}'
        
        # compute the num of completed query in results_dir
        completed_query_num = 0
        for qid in os.listdir(results_dir):
            if os.path.exists(os.path.join(results_dir, qid)):
                completed_query_num += 1
        print(f"{method} completed query num: {completed_query_num}")
        
        if os.path.exists(eval_dir) and os.path.exists(results_dir):
            info = analyze_method(method, results_dir, eval_dir, query_data)
            methods_info[method] = info
        else:
            print(f"\n跳过 {method}: 目录不存在")
    
    # ===== Cross-method comparison =====
    if len(methods_info) >= 2:
        print(f"\n{'='*80}")
        print(f"  跨方法对比分析")
        print(f"{'='*80}")
        
        method_names = list(methods_info.keys())
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                m1, m2 = method_names[i], method_names[j]
                info1, info2 = methods_info[m1], methods_info[m2]
                
                # Common failures
                common_schema_fail = info1['schema_fail'] & info2['schema_fail']
                common_logi_fail = info1['logi_fail'] & info2['logi_fail']
                common_timeout = info1['timeout_ids'] & info2['timeout_ids']
                
                print(f"\n  {m1} vs {m2}:")
                print(f"    共同schema失败: {len(common_schema_fail)}")
                print(f"    共同逻辑约束失败: {len(common_logi_fail)}")
                print(f"    共同超时: {len(common_timeout)}")
                
                # Questions one passes but the other fails
                only_m1_pass = info1['all_pass_ids'] - info2['all_pass_ids']
                only_m2_pass = info2['all_pass_ids'] - info1['all_pass_ids']
                print(f"    仅{m1}通过: {len(only_m1_pass)}")
                print(f"    仅{m2}通过: {len(only_m2_pass)}")

    # ===== Summary of weaknesses =====
    print(f"\n{'='*80}")
    print(f"  Agent缺点总结")
    print(f"{'='*80}")
    
    if 'LLMNeSy_deepseek' in methods_info:
        info = methods_info['LLMNeSy_deepseek']
        total = len(info['all_ids'])
        
        print(f"\n  [最佳Agent: LLMNeSy_deepseek]")
        print(f"  1. Schema通过率不高 (≈62%): 约38%的问题连基本输出格式都不正确")
        print(f"     - 主要原因: 搜索超时 ({len(info['timeout_ids'])}个问题超时)")
        print(f"  2. 逻辑约束满足率低 (macro ≈25%): 即使输出格式正确，仍有大量逻辑约束违反")
        
        # Analyze first constraint (logic_0) failure rate
        logic_0_fail = sum(1 for qid in info['all_ids'] 
                          if info['logical_detail'].get(qid, (0, []))[1] 
                          and 0 in info['logical_detail'].get(qid, (0, []))[1])
        print(f"  3. 第一个逻辑约束(通常是预算约束)失败率特别高: {logic_0_fail}个问题")
        print(f"  4. 约束越多，通过率越低 - agent难以同时满足多个约束")


if __name__ == '__main__':
    main()
