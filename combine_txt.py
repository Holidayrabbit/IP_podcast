import os
import argparse
from pathlib import Path

def combine_txt_files(input_dir, output_file):
    """
    合并指定目录下所有txt文件的内容到一个新的文件中
    
    Args:
        input_dir (str): 输入目录路径
        output_file (str): 输出文件路径
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：目录 '{input_dir}' 不存在")
        return

    # 获取所有txt文件
    txt_files = list(Path(input_dir).glob('*.txt'))
    
    if not txt_files:
        print(f"警告：在目录 '{input_dir}' 中没有找到txt文件")
        return
    
    # 合并文件内容
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for txt_file in sorted(txt_files):
            print(f"正在处理文件: {txt_file}")
            outfile.write(f"\n--- 来自文件: {txt_file.name} ---\n\n")
            
            try:
                with open(txt_file, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                outfile.write('\n')
            except Exception as e:
                print(f"处理文件 '{txt_file}' 时出错: {str(e)}")

    print(f"\n合并完成！输出文件: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='合并指定目录下的所有txt文件')
    parser.add_argument('--input_dir', 
                       default='./output/podcast_scripts39',
                       help='输入目录路径')
    parser.add_argument('--output_file', 
                       default='./output/podcast_scripts39/phone_addiction_topic_podcast_all.txt',
                       help='输出文件路径')
    
    args = parser.parse_args()
    combine_txt_files(args.input_dir, args.output_file)

if __name__ == '__main__':
    main() 