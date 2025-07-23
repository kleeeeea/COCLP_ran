import argparse
import os
import zipfile
import rarfile
import struct
from tqdm import tqdm
import concurrent.futures
import threading
import shutil
import logging
import sys
import datetime
import py7zr

# 添加线程锁，防止多线程写入文件时发生冲突
file_lock = threading.Lock()


def setup_logging(root_path):
    """设置日志记录"""
    # 创建日志目录
    log_dir = os.path.join(root_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件名，包含时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"unzip_log_{timestamp}.txt")

    # 配置日志
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for extraction (default: 4)"
    )

    return parser.parse_args()


def is_zip_encrypted(zip_path):
    with zipfile.ZipFile(zip_path) as zip_file:
        for file_info in zip_file.infolist():
            try:
                with zip_file.open(file_info.filename) as file:
                    file.read(1)
                return False
            except RuntimeError as e:
                print(e)
                if 'encrypted' in str(e):
                    return True
        return False


def is_rar_encrypted(file_path):
    try:
        with rarfile.RarFile(file_path) as rf:
            rf.infolist()
            return False
    except rarfile.NeedPassword as e:
        print(e)
        return True


def is_7z_encrypted(file_path):
    try:
        with py7zr.SevenZipFile(file_path, mode='r') as z:
            if z.needs_password():
                return True
            return False
    except py7zr.exceptions.PasswordRequired:
        return True
    except Exception as e:
        logging.error(f"检查7z加密状态出错: {file_path}, 错误: {str(e)}")
        return False  # 默认返回未加密，让解压流程处理异常


def write_password(file, root_path):
    with file_lock:
        with open(f'{root_path}/unzip_password.txt', 'a+', encoding='utf-8') as f:
            f.write(file)
            f.write('\n')


def write_failed(file, root_path, failed_filename=None):
    with file_lock:
        with open(f'{root_path}/unzip_failed.txt', 'a+', encoding='utf-8') as f:
            f.write(file)
            if failed_filename:
                f.write(f" -> 失败文件: {failed_filename}")
            f.write('\n')


def write_processed(file, root_path):
    with file_lock:
        with open(f'{root_path}/unzip_processed.txt', 'a+', encoding='utf-8') as f:
            f.write(file)
            f.write('\n')


def unzip_single_file(file, root_path):
    """
    args:
        file: 当前压缩包路径
        root_path: 解压的最顶层路径，用来保存失败成果或需要密码的
    """
    # 输出正在解压的文件路径
    logging.info(f"正在解压文件: {file}")

    if file.endswith('.zip'):
        try:
            # 判断 zip 是否需要密码
            if is_zip_encrypted(file):
                logging.info(f"文件需要密码: {file}")
                write_password(file, root_path)
                return False, "需要密码"

            extract_path = os.path.splitext(file)[0]

            if os.path.exists(os.path.splitext(file)[0]):
                extract_path = extract_path + '_unzip'

            logging.info(f"解压到路径: {extract_path}")

            with zipfile.ZipFile(file, 'r') as zip_ref:
                sum1 = 0
                sum2 = 0
                for file_info in zip_ref.infolist():
                    sum1 += 1
                    for encoding in ['gbk', 'utf-8', 'cp437']:
                        try:
                            file_info.filename = file_info.filename.encode('cp437').decode(encoding)
                            zip_ref.extract(file_info, extract_path)
                            sum2 += 1
                            break
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            # print(e)
                            try:
                                file_info.filename = file_info.filename.encode('utf-8').decode(encoding)
                                zip_ref.extract(file_info, extract_path)
                                sum2 += 1
                                break
                            except (UnicodeDecodeError, UnicodeEncodeError) as e:
                                # print(e)
                                continue
                if sum1 == sum2:
                    logging.info(f"解压成功: {file}")
                    write_processed(file, root_path)
                    return True, None
                else:
                    logging.info(f"部分文件解压失败: {file}")
                    # 记录失败的文件名称
                    failed_files = []
                    for file_info in zip_ref.infolist():
                        try:
                            # 尝试获取文件名
                            filename = file_info.filename
                            # 尝试打开文件验证是否可以访问
                            zip_ref.getinfo(filename)
                        except Exception:
                            failed_files.append(filename)

                    if failed_files:
                        for failed_file in failed_files:
                            write_failed(file, root_path, failed_file)
                            logging.info(f"  -- 解压失败的文件: {failed_file}")
                    else:
                        write_failed(file, root_path)
                    return False, "部分文件解压失败"
        except Exception as e:
            # 解压失败记录
            logging.error(f"解压失败: {file}, 错误: {e}")
            write_failed(file, root_path)
            return False, str(e)

    if file.endswith('.rar'):
        try:
            # 判断 rar 是否需要密码
            if is_rar_encrypted(file):
                logging.info(f"文件需要密码: {file}")
                write_password(file, root_path)
                return False, "需要密码"

            extract_path = os.path.splitext(file)[0]

            if os.path.exists(os.path.splitext(file)[0]):
                extract_path = extract_path + '_unzip'

            logging.info(f"解压到路径: {extract_path}")

            rf = rarfile.RarFile(file)
            flag = False
            failed_files = []

            for encoding in ['gbk', 'utf-8', 'cp437']:
                try:
                    rf.encoding = encoding
                    # 先尝试提取文件列表
                    for rarinfo in rf.infolist():
                        try:
                            rf.extract(rarinfo, path=extract_path)
                        except Exception as e:
                            failed_files.append(rarinfo.filename)
                            logging.info(f"  -- 解压失败的文件: {rarinfo.filename}")

                    if not failed_files:
                        logging.info(f"解压成功: {file}")
                        write_processed(file, root_path)
                        flag = True
                        return True, None
                    else:
                        logging.info(f"部分文件解压失败: {file}")
                        for failed_file in failed_files:
                            write_failed(file, root_path, failed_file)
                        flag = True
                        return False, "部分文件解压失败"
                    break
                except Exception:
                    continue
            if not flag:
                logging.info(f"解压失败: {file}")
                write_failed(file, root_path)
                return False, "解压失败"
        except Exception as e:
            logging.error(f"解压失败: {file}, 错误: {e}")
            write_failed(file, root_path)
            return False, str(e)

    if file.endswith('.7z'):
        try:
            # 判断7z文件是否需要密码
            if is_7z_encrypted(file):
                logging.info(f"文件需要密码: {file}")
                write_password(file, root_path)
                return False, "需要密码"

            extract_path = os.path.splitext(file)[0]

            if os.path.exists(os.path.splitext(file)[0]):
                extract_path = extract_path + '_unzip'

            logging.info(f"解压到路径: {extract_path}")

            # 确保目标目录存在
            os.makedirs(extract_path, exist_ok=True)

            # 使用py7zr解压文件
            with py7zr.SevenZipFile(file, mode='r') as z:
                # 尝试解压所有文件
                try:
                    z.extractall(path=extract_path)
                    logging.info(f"解压成功: {file}")
                    write_processed(file, root_path)
                    return True, None
                except Exception as e:
                    logging.error(f"解压7z文件失败: {file}, 错误: {str(e)}")
                    write_failed(file, root_path)
                    return False, str(e)

        except Exception as e:
            logging.error(f"处理7z文件失败: {file}, 错误: {str(e)}")
            write_failed(file, root_path)
            return False, str(e)

    return False, "未知文件类型"


def unzip(path, root_path, max_workers=4):
    """
    args:
        path: 当前需要解压的路径
        root_path: 根目录路径
        max_workers: 最大并行工作线程数
    """
    files = os.listdir(path)

    # 过滤出压缩文件
    compress_files = []
    for file in files:
        if file.startswith('._'):
            continue
        if not file.endswith('zip') and not file.endswith('rar') and not file.endswith('7z'):
            continue
        compress_files.append(os.path.join(path, file))

    # 记录解压结果
    results = {}

    # 使用线程池并行处理压缩文件
    if compress_files:
        logging.info(f"开始并行处理目录 {path} 中的 {len(compress_files)} 个压缩文件")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务到线程池
            future_to_file = {executor.submit(unzip_single_file, file, root_path): file for file in compress_files}

            # 使用tqdm显示进度
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(future_to_file),
                               desc=f"解压 {path}"):
                file = future_to_file[future]
                try:
                    success, reason = future.result()
                    results[file] = (success, reason)
                except Exception as e:
                    logging.error(f"处理文件出错: {file}, 错误: {str(e)}")
                    results[file] = (False, str(e))

    # 遍历所有子目录递归处理
    subdirs = []
    for file in files:
        tmp_path = os.path.join(path, file)
        if os.path.isdir(tmp_path):
            subdirs.append(tmp_path)

    # 递归处理子目录
    sub_results = {}
    for subdir in subdirs:
        sub_res = unzip(subdir, root_path, max_workers)
        sub_results.update(sub_res)

    # 合并结果
    results.update(sub_results)
    return results


def process_results(results, root_path):
    """处理解压结果，删除成功的压缩包，移动失败的压缩包到失败目录"""
    failed_dir = os.path.join(root_path, 'unzip_failed')
    password_dir = os.path.join(root_path, 'unzip_password')

    # 创建失败和密码目录
    os.makedirs(failed_dir, exist_ok=True)
    os.makedirs(password_dir, exist_ok=True)

    success_count = 0
    password_count = 0
    failed_count = 0

    for file, (success, reason) in results.items():
        if success:
            # 删除成功解压的压缩包
            try:
                os.remove(file)
                logging.info(f"已删除成功解压的压缩包: {file}")
                success_count += 1
            except Exception as e:
                logging.error(f"删除压缩包失败: {file}, 错误: {str(e)}")
        elif reason == "需要密码":
            # 移动需要密码的压缩包到密码目录
            try:
                filename = os.path.basename(file)
                dest_path = os.path.join(password_dir, filename)
                shutil.move(file, dest_path)
                logging.info(f"已移动需要密码的压缩包: {file} -> {dest_path}")
                password_count += 1
            except Exception as e:
                logging.error(f"移动压缩包失败: {file}, 错误: {str(e)}")
        else:
            # 移动失败的压缩包到失败目录
            try:
                filename = os.path.basename(file)
                dest_path = os.path.join(failed_dir, filename)
                shutil.move(file, dest_path)
                logging.info(f"已移动解压失败的压缩包: {file} -> {dest_path}")
                failed_count += 1
            except Exception as e:
                logging.error(f"移动压缩包失败: {file}, 错误: {str(e)}")

    return success_count, password_count, failed_count

class UNZIP:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": " ",
                    "lazy": True
                }),
                "max_workers": ("INT", {
                    "default": 1,
                    "min": 1, #Minimum value
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                }),
                "root_path": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": " ",
                    "lazy": True
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)

    FUNCTION = "test"

    OUTPUT_NODE = True

    CATEGORY = "MY_NODES/Tool"

    def test(self, path, root_path, max_workers):
         # 设置日志
        logger = setup_logging(root_path)

        logging.info(f"开始解压目录: {root_path} (使用 {max_workers} 个并行工作线程)")

        # 执行解压并获取结果
        results = unzip(path, root_path, max_workers)

        # 处理结果
        success_count, password_count, failed_count = process_results(results, root_path)

        logging.info("解压处理完成")
        logging.info(f"成功解压并删除: {success_count} 个文件")
        logging.info(f"需要密码并移动: {password_count} 个文件")
        logging.info(f"解压失败并移动: {failed_count} 个文件")
        logging.shutdown()
        print("解压完成")
        return(path,)


