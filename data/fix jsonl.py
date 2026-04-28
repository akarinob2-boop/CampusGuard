import json
import os


def fix_jsonl_file(file_path):
    """
    修复 JSONL 文件中的类型不一致问题
    """
    fixed_lines = []
    errors = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # 修复 id 字段，确保类型一致
                if 'id' in data:
                    # 将所有 id 转换为字符串（或根据你的需求选择一种类型）
                    data['id'] = str(data['id'])

                fixed_lines.append(json.dumps(data, ensure_ascii=False))

            except json.JSONDecodeError as e:
                errors.append(f"行 {i}: JSON解析错误 - {e}")
                print(f"行 {i} 有错误: {line[:100]}...")
                continue

    if errors:
        print(f"发现 {len(errors)} 个错误:")
        for error in errors:
            print(error)

    # 保存修复后的文件
    fixed_path = file_path.replace('.jsonl', '_fixed.jsonl')
    with open(fixed_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))

    print(f"已修复文件: {fixed_path}")
    print(f"原始行数: {i}, 修复后行数: {len(fixed_lines)}")

    return fixed_path


# 修复文件
file_path = "E:\\ai\\data\\val.jsonl"
fixed_file = fix_jsonl_file(file_path)