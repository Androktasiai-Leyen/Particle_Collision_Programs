#!/usr/bin/env python3
"""
坐标转换工具：将events_*.txt中的px py pz转换为pt phi eta
输入格式：event_id particle_id particle_type px py pz
输出格式：event_id particle_id particle_type pt phi eta
"""

import numpy as np
from tqdm import tqdm
import math
import os
import glob

def px_py_pz_to_pt_phi_eta(px, py, pz):
    """
    将笛卡尔坐标(px, py, pz)转换为球坐标(pt, phi, eta)
    
    参数:
    px, py, pz: 笛卡尔动量分量
    
    返回:
    pt: 横向动量
    phi: 方位角 [0, 2π]
    eta: 赝快度
    """
    # 计算横向动量
    pt = math.sqrt(px**2 + py**2)
    
    # 计算方位角
    if pt > 0:
        phi = math.atan2(py, px)
        # 确保phi在[0, 2π]范围内
        if phi < 0:
            phi += 2 * math.pi
    else:
        phi = 0.0
    
    # 计算赝快度
    if pz != 0:
        # 计算总动量
        p_total = math.sqrt(px**2 + py**2 + pz**2)
        if p_total > 0:
            # 计算cos(theta)
            cos_theta = pz / p_total
            # 限制cos_theta在[-1, 1]范围内
            cos_theta = max(-1.0, min(1.0, cos_theta))
            # 计算赝快度
            eta = -0.5 * math.log((1.0 - cos_theta) / (1.0 + cos_theta))
        else:
            eta = 0.0
    else:
        eta = 0.0
    
    return pt, phi, eta

def convert_single_file(input_file, output_file):
    """
    转换单个文件的坐标格式
    """
    print(f"🔄 转换文件: {input_file} -> {output_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件 {input_file} 不存在!")
        return False, 0, 0
    
    total_lines = 0
    converted_lines = 0
    error_lines = 0
    
    # 首先计算总行数
    with open(input_file, 'r') as f:
        total_lines = sum(1 for line in f)
    
    # 删除已存在的输出文件
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # 开始转换
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 6:
                try:
                    # 提取前三列（保持不变）
                    event_id = parts[0]
                    particle_id = parts[1]
                    particle_type = parts[2]
                    
                    # 提取px, py, pz
                    px = float(parts[3])
                    py = float(parts[4])
                    pz = float(parts[5])
                    
                    # 转换为pt, phi, eta
                    pt, phi, eta = px_py_pz_to_pt_phi_eta(px, py, pz)
                    
                    # 写入新格式
                    new_line = f"{event_id} {particle_id} {particle_type} {pt:.6f} {phi:.6f} {eta:.6f}\n"
                    outfile.write(new_line)
                    converted_lines += 1
                    
                except (ValueError, IndexError) as e:
                    error_lines += 1
                    continue
            else:
                error_lines += 1
                continue
    
    return True, converted_lines, error_lines

def convert_all_events_files(input_pattern="Mevents_*.txt", output_dir="converted_events"):
    """
    转换所有Mevents_*.txt文件
    """
    print(f"🔄 开始转换所有Mevents_*.txt文件...")
    print(f"📁 输入文件模式: {input_pattern}")
    print(f"📁 输出目录: {output_dir}")
    
    # 在pythia8315/examples目录中查找文件
    events_dir = "pythia8315/examples"
    if os.path.exists(events_dir):
        input_pattern = os.path.join(events_dir, input_pattern)
        print(f"📁 在目录 {events_dir} 中查找文件")
    else:
        print(f"⚠️  目录 {events_dir} 不存在，在当前目录查找")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 获取所有匹配的文件
    input_files = sorted(glob.glob(input_pattern))
    if not input_files:
        print(f"❌ 错误: 没有找到匹配的文件: {input_pattern}")
        return False
    
    print(f"📊 找到 {len(input_files)} 个文件需要转换")
    
    total_converted = 0
    total_errors = 0
    failed_files = []
    
    # 转换每个文件
    for input_file in tqdm(input_files, desc="转换文件"):
        # 生成输出文件名
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"converted_{base_name}")
        
        # 转换文件
        success, converted, errors = convert_single_file(input_file, output_file)
        
        if success:
            total_converted += converted
            total_errors += errors
            print(f"✅ {base_name}: 转换 {converted:,} 行, 错误 {errors:,} 行")
        else:
            failed_files.append(input_file)
            print(f"❌ {base_name}: 转换失败")
    
    # 输出总结
    print(f"\n" + "="*60)
    print(f"📊 转换总结:")
    print(f"📁 总文件数: {len(input_files)}")
    print(f"✅ 成功转换: {len(input_files) - len(failed_files)}")
    print(f"❌ 失败文件: {len(failed_files)}")
    print(f"📊 总转换行数: {total_converted:,}")
    print(f"📊 总错误行数: {total_errors:,}")
    
    if failed_files:
        print(f"\n❌ 失败的文件:")
        for f in failed_files:
            print(f"   {f}")
    
    return True

def verify_conversion(output_dir="converted_events"):
    """
    验证转换结果
    """
    print(f"\n🔍 验证转换结果...")
    
    if not os.path.exists(output_dir):
        print(f"❌ 输出目录 {output_dir} 不存在!")
        return
    
    # 获取所有转换后的文件
    converted_files = glob.glob(os.path.join(output_dir, "converted_Mevents_*.txt"))
    print(f"📁 转换后的文件数量: {len(converted_files)}")
    
    if not converted_files:
        print(f"❌ 没有找到转换后的文件!")
        return
    
    # 显示第一个文件的前几行
    first_file = converted_files[0]
    print(f"\n📋 第一个转换文件 {os.path.basename(first_file)} 的前5行:")
    with open(first_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"   {line.strip()}")
    
    # 统计总文件大小
    total_size = 0
    for file in converted_files:
        total_size += os.path.getsize(file)
    
    print(f"\n📊 总文件大小: {total_size / (1024**3):.2f} GB")

def main():
    print("="*80)
    print("🚀 Events文件坐标转换工具: px py pz → pt phi eta")
    print("="*80)
    
    # 执行转换
    success = convert_all_events_files()
    
    if success:
        # 验证转换结果
        verify_conversion()
        
        print("\n" + "="*60)
        print("✅ 所有文件转换完成！")
        print("="*60)
        print(f"📁 输出目录: converted_events/")
        print(f"📊 格式: event_id particle_id particle_type pt phi eta")
    else:
        print("\n❌ 转换失败！")

if __name__ == "__main__":
    main() 