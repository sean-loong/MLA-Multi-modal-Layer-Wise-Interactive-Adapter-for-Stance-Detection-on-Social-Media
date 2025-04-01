import pandas as pd

def process_path(path):
    """处理图片路径并生成新路径"""
    # 统一路径分隔符并提取文件名
    filename = path.replace('\\', '/').split('/')[-1]
    return f'images/{filename}'

def process_dataset(input_file, output_file):
    """处理单个数据集文件"""
    # 读取原始数据
    df = pd.read_csv(input_file)
    
    # 将空字符串转换为NaN以便正确识别
    df['tweet_images'] = df['tweet_images'].replace('', pd.NA)
    df['tweet_video_frames'] = df['tweet_video_frames'].replace('', pd.NA)
    
    # 合并图片列（优先取tweet_images，空值时用tweet_video_frames）
    combined_path = df['tweet_images'].combine_first(df['tweet_video_frames'])
    
    # 过滤两列都为空的行（保留combined_path不为空的行）
    valid_mask = combined_path.notna()
    df = df[valid_mask].copy()
    combined_path = combined_path[valid_mask]
    
    # 生成新路径列
    df['new_images'] = combined_path.apply(process_path)
    
    # 创建新数据集并保持原始列顺序
    new_df = df[['tweet_id', 'tweet_text', 'stance_target', 'stance_label']].copy()
    new_df['tweet_image'] = df['new_images']
    
    # 保存处理后的文件
    new_df.to_csv(output_file, index=False)

# 处理三个数据集
datasets = ['train', 'test', 'valid']
for dataset in datasets:
    process_dataset(f'original/{dataset}.csv', f'{dataset}.csv')