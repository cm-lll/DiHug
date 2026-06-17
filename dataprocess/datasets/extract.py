import os

def take_first_jsonl(input_file, output_dir, output_name="subset.jsonl", num_lines=100):
    """
    从 jsonl 文件中取前 num_lines 行，保存到指定目录。
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_name)

    with open(input_file, "r", encoding="utf8") as fin, \
         open(output_file, "w", encoding="utf8") as fout:
        for i, line in enumerate(fin):
            if i >= num_lines:
                break
            fout.write(line.strip() + "\n")

    print(f"[INFO] 已从 {input_file} 取前 {num_lines} 行，保存到 {output_file}")


# 使用示例
if __name__ == "__main__":
    take_first_jsonl(
        input_file="ACM-Citation-network-V12.jsonl",       # 原始文件路径
        output_dir="../preprocess/data/aminer_raw",             # 输出目录
        output_name="aminer_subset.jsonl",    # 输出文件名
        num_lines=1000                         # 取前 n 行
    )
