#!/usr/bin/env python3
"""
按照5个区间划分多重度事件
从converted_events_*.txt文件中提取数据，按多重度分为：
0-20%（最低），20%-40%，40%-60%，60%-80%，80-100%（最高）
"""

import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
import glob

def count_multiplicity_per_event(file_path):
    """统计每个事件的粒子数量（多重度）"""
    print(f"📊 分析文件: {os.path.basename(file_path)}")
    
    event_multiplicity = defaultdict(int)
    total_lines = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 6:
                    event_id = parts[0]
                    event_multiplicity[event_id] += 1
                    total_lines += 1
    
    print(f"   📈 总行数: {total_lines:,}")
    print(f"   📊 事件数: {len(event_multiplicity):,}")
    
    return event_multiplicity

def get_all_multiplicities():
    """获取所有converted_events文件的事件多重度"""
    print("🔍 开始分析所有converted_events文件的多重度分布...")
    
    all_multiplicities = {}
    
    # 获取所有converted_Mevents文件
    converted_files = sorted(glob.glob("converted_events/converted_Mevents_*.txt"))
    print(f"📁 找到 {len(converted_files)} 个converted_Mevents文件")
    
    for file_path in tqdm(converted_files, desc="分析文件"):
        file_multiplicities = count_multiplicity_per_event(file_path)
        
        # 为每个事件添加文件标识符
        for event_id, multiplicity in file_multiplicities.items():
            file_index = int(file_path.split('_')[-1].split('.')[0])
            unique_event_id = f"{file_index}_{event_id}"
            all_multiplicities[unique_event_id] = {
                'multiplicity': multiplicity,
                'file_path': file_path,
                'original_event_id': event_id
            }
    
    return all_multiplicities

def find_multiplicity_percentiles(all_multiplicities):
    """找出5个区间的多重度分位数"""
    print("\n🎯 计算多重度分布和分位数...")
    
    multiplicities = [data['multiplicity'] for data in all_multiplicities.values()]
    multiplicities = np.array(multiplicities)
    
    print(f"📊 多重度统计:")
    print(f"   平均值: {np.mean(multiplicities):.2f}")
    print(f"   标准差: {np.std(multiplicities):.2f}")
    print(f"   最小值: {np.min(multiplicities)}")
    print(f"   最大值: {np.max(multiplicities)}")
    
    # 计算5个区间的分位数
    percentiles = [20, 40, 60, 80]
    thresholds = [np.percentile(multiplicities, p) for p in percentiles]
    
    print(f"📊 分位数:")
    print(f"   20%分位数: {thresholds[0]:.2f}")
    print(f"   40%分位数: {thresholds[1]:.2f}")
    print(f"   60%分位数: {thresholds[2]:.2f}")
    print(f"   80%分位数: {thresholds[3]:.2f}")
    
    # 按区间分组事件
    interval_events = {
        '0-20%': [],    # 最低
        '20-40%': [],   # 低
        '40-60%': [],   # 中等
        '60-80%': [],   # 高
        '80-100%': []   # 最高
    }
    
    for unique_event_id, data in all_multiplicities.items():
        multiplicity = data['multiplicity']
        
        if multiplicity < thresholds[0]:
            interval_events['0-20%'].append(data)
        elif multiplicity < thresholds[1]:
            interval_events['20-40%'].append(data)
        elif multiplicity < thresholds[2]:
            interval_events['40-60%'].append(data)
        elif multiplicity < thresholds[3]:
            interval_events['60-80%'].append(data)
        else:
            interval_events['80-100%'].append(data)
    
    # 打印每个区间的事件数量
    total_events = len(all_multiplicities)
    for interval, events in interval_events.items():
        percentage = len(events) / total_events * 100
        print(f"   {interval}: {len(events):,} 事件 ({percentage:.1f}%)")
    
    return interval_events, thresholds

def extract_interval_data(interval_events):
    """提取每个区间的事件粒子数据"""
    print("\n📝 开始提取各区间事件数据...")
    
    interval_outputs = {}
    
    for interval, events in tqdm(interval_events.items(), desc="处理区间"):
        print(f"\n🔍 处理区间: {interval}")
        
        output_lines = []
        processed_events = 0
        
        # 按文件分组处理
        file_groups = defaultdict(list)
        for event_data in events:
            file_groups[event_data['file_path']].append(event_data['original_event_id'])
        
        for file_path, event_ids in tqdm(file_groups.items(), desc=f"处理{interval}区间"):
            event_ids_set = set(event_ids)
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            event_id = parts[0]
                            if event_id in event_ids_set:
                                output_lines.append(line.strip())
            
            processed_events += len(event_ids)
        
        print(f"   📊 提取了 {len(output_lines):,} 行数据")
        print(f"   📊 涉及 {processed_events:,} 个事件")
        
        interval_outputs[interval] = output_lines
    
    return interval_outputs

def save_interval_files(interval_outputs):
    """为每个区间保存单独的文件"""
    print(f"\n💾 保存各区间数据到文件...")
    
    saved_files = {}
    
    for interval, output_lines in interval_outputs.items():
        # 生成文件名，使用ff+前缀，将百分号替换为下划线
        filename = f"ff+{interval.replace('%', '').replace('-', '_')}.txt"
        
        print(f"   💾 保存 {interval} 区间到 {filename}...")
        
        with open(filename, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        
        print(f"   ✅ 成功保存 {len(output_lines):,} 行数据到 {filename}")
        saved_files[interval] = filename
    
    return saved_files

def main():
    print("="*80)
    print("🚀 多重度5区间事件提取工具")
    print("📁 数据源: converted_events/ 目录")
    print("🎯 区间: 0-20%（最低）, 20-40%, 40-60%, 60-80%, 80-100%（最高）")
    print("📝 输出文件前缀: ff+")
    print("="*80)
    
    # 第一步：获取所有事件的多重度
    all_multiplicities = get_all_multiplicities()
    
    # 第二步：找出5个区间的多重度分位数和事件分组
    interval_events, thresholds = find_multiplicity_percentiles(all_multiplicities)
    
    # 第三步：提取每个区间的事件数据
    interval_outputs = extract_interval_data(interval_events)
    
    # 第四步：为每个区间保存单独的文件
    saved_files = save_interval_files(interval_outputs)
    
    print("\n" + "="*60)
    print("✅ 提取完成！")
    print("="*60)
    print("📁 输出文件:")
    for interval, filename in saved_files.items():
        print(f"   {interval}: {filename}")
    
    print(f"\n📊 多重度区间阈值:")
    print(f"   0-20%: < {thresholds[0]:.2f}")
    print(f"   20-40%: {thresholds[0]:.2f} - {thresholds[1]:.2f}")
    print(f"   40-60%: {thresholds[1]:.2f} - {thresholds[2]:.2f}")
    print(f"   60-80%: {thresholds[2]:.2f} - {thresholds[3]:.2f}")
    print(f"   80-100%: ≥ {thresholds[3]:.2f}")

if __name__ == "__main__":
    main() 