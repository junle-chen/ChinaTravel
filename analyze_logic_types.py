import sys
sys.path.insert(0, '.')
from chinatravel.data.load_datasets import load_query
import argparse
from collections import Counter
import re

parser = argparse.ArgumentParser()
parser.add_argument('--splits', type=str, default='human')
parser.add_argument('--oracle_translation', action='store_true', default=True)
args = parser.parse_args([])
_, query_data = load_query(args)

logic_patterns = {
    '天数 (day_count)': r'day_count\(plan\)==',
    '人数 (people_count)': r'people_count\(plan\)==',
    '出租车数量合理性 (taxi_cars)': r'taxi_cars\(activity_transports\(activity\)\)!=',
    '特定必去景点 (attraction_name_set)': r"attraction_name_set\.add\(activity_position\(activity\)\)",
    '特定景点类型 (attraction_type_set)': r"attraction_type_set\.add\(attraction_type\(activity",
    '必须包含某种活动类别 (activity_type_set)': r"activity_type_set\.add\(activity_type\(activity\)\)",
    '预算上限 (total_cost)': r'total_cost\s*<=\s*\d+',
    '餐饮菜系 (restaurant_type_set)': r"restaurant_type_set\.add\(restaurant_type\(activity",
    '酒店特色标签 (accommodation_type_set)': r"accommodation_type_set\.add\(accommodation_type\(activity",
    '指定酒店名称 (hotel_name_set)': r"hotel_name_set\.add\(activity_position\(activity\)\)",
    '城际交通方式限制 (intercity_transport_set)': r"intercity_transport_set\.add\(intercity_transport_type\(activity\)\)",
    '市内交通方式限制 (innercity_transport_set)': r"innercity_transport_set\.add\(innercity_transport_type\(activity_transports",
    '出发时间限制 (earliest_start_time)': r"time_compare_if_earlier_equal",
    '到达时间限制 (latest_end_time)': r"activity_end_time\(activity",
    '房型限制 (room_type_set)': r"room_type_set\.add\(accommodation_room_type\(activity",
    '特定餐厅 (restaurant_name_set)': r"restaurant_name_set\.add\(activity_position\(activity\)\)"
}

logic_counts = Counter()
example_code = {}

for qid, q in query_data.items():
    for logic_str in q.get('hard_logic_py', []):
        matched = False
        for logic_name, pattern in logic_patterns.items():
            if re.search(pattern, logic_str):
                logic_counts[logic_name] += 1
                if logic_name not in example_code:
                    example_code[logic_name] = logic_str.strip()
                matched = True
                break
        
        if not matched:
            logic_counts['其他未分类'] += 1
            if '其他未分类' not in example_code:
                example_code['其他未分类'] = logic_str.strip()

print("=== Human Split 中包含的各类 hard_logic_py 统计 ===")
total = sum(logic_counts.values())
for name, count in sorted(logic_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"\n- {name}: 出现 {count} 次 ({count/total*100:.1f}%)")
    print("  示例代码:")
    lines = example_code[name].split('\n')
    for line in lines[:5]:
        print(f"    {line}")
    if len(lines) > 5:
        print("    ...")
