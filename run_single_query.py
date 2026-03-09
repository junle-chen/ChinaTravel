"""
运行单个查询并将完整的思考/搜索过程写入文件。

用法示例:
  # 使用 GPT-5.1 运行 LLMNeSy agent
  python run_single_query.py --query_id h20241029143447759844 --agent LLMNeSy --llm gpt-5.1-chat

  # 使用 deepseek 运行
  python run_single_query.py --query_id h20241029143447759844 --agent LLMNeSy --llm deepseek

  # 使用 oracle translation
  python run_single_query.py --query_id h20241029143447759844 --agent LLMNeSy --llm gpt-5.1-chat --oracle_translation

  # 指定输出目录
  python run_single_query.py --query_id h20241029143447759844 --agent LLMNeSy --llm gpt-5.1-chat --output_dir case_study_output

  # 使用自然语言输入 (命令行指定)
  python run_single_query.py --query_text "我一个人想去北京玩2天..." --agent LLMNeSy --llm gpt-5.1-chat

  # 使用自然语言输入 (直接在脚本中修改 DEFAULT_QUERY_TEXT)
  python run_single_query.py --agent LLMNeSy --llm gpt-5.1-chat

输出:
  - <output_dir>/<query_id>_result.json    : Agent 生成的计划结果
  - <output_dir>/<query_id>_trace.log      : 完整的思考和搜索过程日志
  - <output_dir>/<query_id>_query_info.txt : 查询的基本信息和约束
"""

import argparse
import sys
import os
import json
import time
import io
import hashlib

project_root_path = os.path.dirname(os.path.abspath(__file__))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from copy import deepcopy
from chinatravel.data.load_datasets import load_query, save_json_file
from chinatravel.agent.load_model import init_agent, init_llm
from chinatravel.environment.world_env import WorldEnv


# ==========================================
# 在此处修改默认的自然语言查询 (如果没指定 --query_id 或 --query_text)
# ==========================================
DEFAULT_QUERY_TEXT = "我想从上海去苏州旅游，两个人，三天，想要体验园林"
# ==========================================


class TeeLogger:
    """同时写入文件和终端的 Logger"""
    def __init__(self, filename, terminal):
        self.terminal = terminal
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def write_query_info(query, output_path):
    """将查询信息和约束写入可读文本文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  查询详情\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"  查询 ID:   {query.get('uid', 'N/A')}\n")
        f.write(f"  出发城市:  {query.get('start_city', 'N/A')}\n")
        f.write(f"  目标城市:  {query.get('target_city', 'N/A')}\n")
        f.write(f"  天数:      {query.get('days', 'N/A')}\n")
        f.write(f"  人数:      {query.get('people_number', 'N/A')}\n")
        f.write(f"\n")

        # 自然语言查询
        nl = query.get('nature_language', '')
        if nl:
            f.write(f"  [自然语言查询 (中文)]\n")
            f.write(f"    {nl}\n\n")

        nl_en = query.get('nature_language_en', '')
        if nl_en:
            f.write(f"  [Natural Language Query (English)]\n")
            f.write(f"    {nl_en}\n\n")

        # 逻辑约束
        hard_logic_py = query.get('hard_logic_py', [])
        if hard_logic_py:
            f.write(f"  [Python 逻辑约束] (共 {len(hard_logic_py)} 个)\n")
            for i, py_code in enumerate(hard_logic_py):
                py_clean = py_code.replace('\n', '\n    ')
                f.write(f"    约束 {i}:\n")
                f.write(f"    {py_clean}\n\n")

        # 其他字段
        f.write(f"  [其他参数]\n")
        f.write(f"    limit_rooms:      {query.get('limit_rooms', 'N/A')}\n")
        f.write(f"    limits_room_type: {query.get('limits_room_type', 'N/A')}\n")
        f.write(f"    tag:              {query.get('tag', 'N/A')}\n")


def main():
    parser = argparse.ArgumentParser(description="运行单个查询并记录完整过程")
    parser.add_argument("--query_id", "-q", type=str, default=None,
                        help="查询ID, 例如 h20241029143447759844")
    parser.add_argument("--query_text", "-t", type=str, default=None,
                        help="直接输入自然语言查询内容")
    parser.add_argument("--splits", "-s", type=str, default="human",
                        help="数据集 split (default: human)")
    parser.add_argument("--agent", "-a", type=str, default="LLMNeSy",
                        choices=["RuleNeSy", "LLMNeSy", "LLM-modulo", "ReAct", "ReAct0", "Act", "TPCAgent"],
                        help="Agent 类型 (default: LLMNeSy)")
    parser.add_argument("--llm", "-l", type=str, default="gpt-5.1-chat",
                        help="LLM 名称 (default: gpt-5.1-chat)")
    parser.add_argument("--oracle_translation", action="store_true",
                        help="使用 oracle translation")
    parser.add_argument("--preference_search", action="store_true",
                        help="使用 preference search")
    parser.add_argument("--refine_steps", type=int, default=10,
                        help="LLM-modulo refine 步数")
    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--output_dir", "-o", type=str, default="case_study_output",
                        help="输出目录 (default: case_study_output)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    # ===== 1. 加载或准备查询数据 =====
    query = None
    if args.query_id:
        print(f"加载数据集 splits={args.splits} ...")
        query_index, query_data = load_query(args)
        if args.query_id not in query_data:
            print(f"\n[错误] 查询 ID '{args.query_id}' 不在当前数据集中!")
            sys.exit(1)
        query = query_data[args.query_id]
    else:
        # 使用自然语言输入
        text = args.query_text or DEFAULT_QUERY_TEXT
        if not text:
            print("\n[错误] 请提供 --query_id, --query_text, 或在脚本中设置 DEFAULT_QUERY_TEXT")
            sys.exit(1)
        
        # 生成一个伪 ID
        pseudo_id = "custom_" + hashlib.md5(text.encode('utf-8')).hexdigest()[:10]
        args.query_id = pseudo_id
        
        # 构造基础 query 对象 (部分字段会在后续由 Agent 提取或翻译)
        query = {
            "uid": pseudo_id,
            "nature_language": text,
            "nature_language_en": "", # 后续可能会翻译
            "start_city": None, # 后续提取
            "target_city": None, # 后续提取
            "days": None,           # 后续提取
            "people_number": None,  # 后续提取
            "hard_logic_py": []
        }

    print(f"\n目标查询: {args.query_id}")
    if query.get('start_city'):
        print(f"  {query.get('start_city')} -> {query.get('target_city')}, "
              f"{query.get('days')}天, {query.get('people_number')}人, "
              f"{len(query.get('hard_logic_py', []))}个逻辑约束")
    else:
        print(f"  自然语言: {query.get('nature_language')[:50]}...")

    # ===== 2. 准备输出目录 =====
    output_dir = os.path.join(project_root_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    query_info_path = os.path.join(output_dir, f"{args.query_id}_query_info.txt")
    trace_log_path = os.path.join(output_dir, f"{args.query_id}_trace.log")
    result_path = os.path.join(output_dir, f"{args.query_id}_result.json")

    # 写入查询信息
    write_query_info(query, query_info_path)
    print(f"\n查询信息已写入: {query_info_path}")

    # ===== 3. 初始化 Agent =====
    print(f"\n初始化 Agent: {args.agent} + {args.llm}")

    cache_dir = os.path.join(project_root_path, "cache")
    method = args.agent + "_" + args.llm
    if args.agent == "LLM-modulo":
        method += f"_{args.refine_steps}steps"
    if args.oracle_translation:
        method += "_oracletranslation"
    if args.preference_search:
        method += "_preferencesearch"

    log_dir = os.path.join(project_root_path, "cache", method)
    os.makedirs(log_dir, exist_ok=True)

    max_model_len = 65536
    kwargs = {
        "method": args.agent,
        "env": WorldEnv(),
        "backbone_llm": init_llm(args.llm, max_model_len=max_model_len),
        "cache_dir": cache_dir,
        "log_dir": log_dir,
        "debug": True,
        "refine_steps": args.refine_steps,
    }
    agent = init_agent(kwargs)

    # ===== 4. 执行查询并捕获完整日志 =====
    print(f"\n{'='*60}")
    print(f"  开始执行查询: {args.query_id}")
    print(f"  Agent: {args.agent}, LLM: {args.llm}")
    print(f"  日志输出到: {trace_log_path}")
    print(f"{'='*60}\n")

    start_time = time.time()

    if args.agent in ["ReAct", "ReAct0", "Act"]:
        # ReAct/Act: 手动设置 TeeLogger 捕获输出
        tee_stdout = TeeLogger(trace_log_path, sys.__stdout__)
        try:
            sys.stdout = tee_stdout
            plan_log = agent(query["nature_language"])
            plan = plan_log["ans"]
            if isinstance(plan, str):
                try:
                    plan = json.loads(plan)
                except:
                    plan = {"plan": plan}
            if not isinstance(plan, dict):
                plan = {"plan": plan}
            plan["input_token_count"] = agent.backbone_llm.input_token_count
            plan["output_token_count"] = agent.backbone_llm.output_token_count
            plan["input_token_maxx"] = agent.backbone_llm.input_token_maxx
            succ = True
        finally:
            sys.stdout = sys.__stdout__
            tee_stdout.close()

    elif args.agent == "LLM-modulo":
        tee_stdout = TeeLogger(trace_log_path, sys.__stdout__)
        try:
            sys.stdout = tee_stdout
            succ, plan = agent.solve(query, prob_idx=args.query_id, oracle_verifier=True)
        finally:
            sys.stdout = sys.__stdout__
            tee_stdout.close()

    elif args.agent in ["LLMNeSy", "RuleNeSy"]:
        # NesyAgent.run() 内部会将 sys.stdout 重定向到自己的 Logger
        # Logger 使用 backbone_llm.name (如 "DeepSeek-V3") 而非 args.llm (如 "deepseek")
        # 所以不能预先设置 TeeLogger，只能事后复制 agent 内部的 log 文件
        succ, plan = agent.run(
            query,
            load_cache=True,
            oralce_translation=args.oracle_translation,
            preference_search=args.preference_search
        )

        # 打印提取后的信息 (调试用)
        if not args.query_id or args.query_id.startswith("custom_"):
            print(f"\n[Agent 提取信息]")
            print(f"  城市: {query.get('start_city')} -> {query.get('target_city')}")
            print(f"  天数: {query.get('days')}, 人数: {query.get('people_number')}")
            print(f"  约束数: {len(query.get('hard_logic_py', []))}")

        # 运行完后从 agent 实际使用的 log_dir 复制日志
        import shutil
        actual_log_path = os.path.join(agent.log_dir, f"{args.query_id}.log")
        if os.path.exists(actual_log_path):
            shutil.copy2(actual_log_path, trace_log_path)
            print(f"Agent 日志已复制: {actual_log_path} -> {trace_log_path}")
        else:
            print(f"[警告] Agent 日志文件不存在: {actual_log_path}")
            print(f"  提示: Agent 实际 log_dir = {agent.log_dir}")

    elif args.agent == "TPCAgent":
        tee_stdout = TeeLogger(trace_log_path, sys.__stdout__)
        try:
            sys.stdout = tee_stdout
            succ, plan = agent.run(query, prob_idx=args.query_id, oralce_translation=args.oracle_translation)
        finally:
            sys.stdout = sys.__stdout__
            tee_stdout.close()

    elapsed = time.time() - start_time

    # ===== 5. 保存结果 =====
    save_json_file(json_data=plan, file_path=result_path)

    # ===== 6. 打印摘要 =====
    print(f"\n{'='*60}")
    print(f"  执行完成!")
    print(f"{'='*60}")
    print(f"  查询 ID:      {args.query_id}")
    print(f"  成功:          {'是' if succ else '否'}")
    print(f"  耗时:          {elapsed:.1f} 秒")
    print(f"  有 itinerary:  {'是' if 'itinerary' in plan else '否'}")

    if 'search_nodes' in plan:
        print(f"  搜索节点数:    {plan['search_nodes']}")
    if 'backtrack_count' in plan:
        print(f"  回溯次数:      {plan['backtrack_count']}")
    if 'llm_rec_count' in plan:
        print(f"  LLM 调用次数:  {plan['llm_rec_count']}")
    if 'llm_rec_format_error_count' in plan:
        print(f"  LLM 格式错误:  {plan['llm_rec_format_error_count']}")
    if 'constraints_validation_count' in plan:
        print(f"  约束验证次数:  {plan['constraints_validation_count']}")
    if 'commonsense_pass_count' in plan:
        print(f"  常识通过次数:  {plan['commonsense_pass_count']}")
    if 'logical_pass_count' in plan:
        print(f"  逻辑通过次数:  {plan['logical_pass_count']}")
    if plan.get('time_out_flag'):
        print(f"  [!] 搜索超时!")

    print(f"\n  输出文件:")
    print(f"    查询信息:  {query_info_path}")
    print(f"    搜索日志:  {trace_log_path}")
    print(f"    结果 JSON: {result_path}")


if __name__ == "__main__":
    main()
