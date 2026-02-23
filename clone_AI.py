# ライブラリのインポート
import cv2 
import numpy as np 
import time
import os
from collections import defaultdict
import pickle
import psutil
import csv
import insightface
from insightface.app import FaceAnalysis
import pynvml

# 関数定義

# コサイン類似度計算 - L2正規化済みなのでただの内積
def cosine_sim(a, b):
    return np.dot(a, b)

def gpu_memory_total_used():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    pynvml.nvmlShutdown()
    return mem.used / (1024 * 1024)

def process(video_path):
    cap = cv2.VideoCapture(video_path)
    print(f"動画ファイルを取得中 : {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total frames : {total_frames}")

    # mp4出力する際
    wait = 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # mp4書き込み設定
    base = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{base}_output.mp4"
     # mp4v - MPEG-4 Video Codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("error : VideoWriterが初期化できませんでした")

    # csv書き込み設定
    csv_path = f"performance_log_{base}.csv"

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "process_time_ms",
        "fps",
        "cpu_percent / core",
        "gpu_memory_mb",
        "memory_mb",
        "match_rate"
    ])

    tpf_list = []
    fps_list = []
    cpu_list = []
    ram_list = []
    process_time_list = []
    gpu_memory_list = []
    frame_num = 0
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)
    detectedframe = 0
    detectedunknownframe = 0
    # メイン処理
    while True:
        start_time = time.perf_counter()
        ret, frame = cap.read() 
        # 映像が取得できなかった場合はループを抜ける
        if frame_num == 0:
            if not ret:
                print(f"動画ファイルを取得できません : {video_path}") 
                break 
            else:
                print(f"動画ファイルを取得しました : {video_path}")
    
        frame_num += 1
        if not ret:
            print("完了")
            break
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #顔の領域の検出
            faces = arcface.get(rgb)
            #assert len(input_frame_location) != 0, "画像から顔の検出に失敗しました"
    
            if len(faces) == 0:
                # 顔が見つからないフレームはスキップして次フレームへ
        
                end_time = time.perf_counter()
                process_time = end_time - start_time
                #print(f"処理時間: {process_time*1000:.2f} ms")
                fps = 1 / process_time
                logical_cores = psutil.cpu_count(logical=True)
                cpu_percent = process.cpu_percent(interval=None)
                cpu = cpu_percent / logical_cores
                mem = process.memory_info().rss / (1024 * 1024)
                match_rate = "N/A"
                gpu_memory = gpu_memory_total_used()
                #print()
                #print(f"推定FPS: {fps:.2f}")
                #print(f"CPU使用率: {cpu:.2f}%")
                #print(f"メモリ使用量: {mem:.2f} MB")
                #print(f"一致率: {match_rate}")
                csv_writer.writerow([
                    frame_num,
                    process_time * 1000,
                    fps,
                    cpu,
                    gpu_memory,
                    mem,
                    match_rate
                ])
                writer.write(frame)
                tpf_list.append(process_time*1000)
                fps_list.append(fps)
                cpu_list.append(cpu)
                ram_list.append(mem)
                gpu_memory_list.append(gpu_memory)
                process_time_list.append(process_time * 1000)
                continue

            # 特徴量を抽出する
            for face in faces:
                embedding = face.normed_embedding # normed_embedding - 正規化された特徴量ベクトル
                x1 , y1 , x2 , y2 = face.bbox.astype(int) # bbox - 顔のバウンディングボックス座標
                best_sim = -1.0
                best_name = None
                for name, avg_emb in avg_encodings.items():
                    sim = cosine_sim(avg_emb, embedding)
                    # 最近傍
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name
                # 閾値0.48
                if best_sim > 0.48:
                        name = best_name
                        color = (0, 255, 0)
                        match_rate = f" ({best_sim*100:.1f}%)"
                        detectedframe += 1
                else:
                        name = "unknown"
                        color = (0, 0, 255)
                        match_rate = f" ({best_sim*100:.1f}%)"
                        detectedunknownframe += 1

                # 枠と名前表示
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 
                cv2.putText(frame, name+match_rate, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # mp4出力
            writer.write(frame)

            end_time = time.perf_counter()
            process_time = end_time - start_time
            #print(f"処理時間: {process_time*1000:.2f} ms")
            fps = 1 / process_time
            logical_cores = psutil.cpu_count(logical=True)
            cpu_percent = process.cpu_percent(interval=None)
            cpu = cpu_percent / logical_cores
            gpu_memory = gpu_memory_total_used()
            mem = process.memory_info().rss / (1024 * 1024)
            #print()
            #print(f"推定FPS: {fps:.2f}")
            #print(f"CPU使用率: {cpu:.2f}%")
            #print(f"メモリ使用量: {mem:.2f} MB")
            #print(f"一致率: {match_rate}")
            csv_writer.writerow([
                frame_num,
                process_time * 1000,
                fps,
                cpu,
                gpu_memory,
                mem,
                match_rate
            ])
            tpf_list.append(process_time*1000)
            fps_list.append(fps)
            cpu_list.append(cpu)
            ram_list.append(mem)
            gpu_memory_list.append(gpu_memory)
            process_time_list.append(process_time * 1000)

    # ウィンドウの解放
    cap.release() 
    cv2.destroyAllWindows() 
    csv_file.close()
    writer.release()
    print(f"\n動画を保存しました : {output_path}")
    print(f"CSVログを保存しました : {csv_path}")
    avg_tpf = avg(tpf_list)
    avg_fps = avg(fps_list)
    avg_cpu = avg(cpu_list)
    avg_ram = avg(ram_list)
    avg_gpu_memory = avg(gpu_memory_list)

    false_negative_rate = (detectedunknownframe / (detectedframe + detectedunknownframe))*100 if (detectedframe + detectedunknownframe) > 0 else 0
    detected_rate = (detectedframe + detectedunknownframe)/total_frames*100
    true_detection_rate = (detectedframe/total_frames)*100

    txt_path = f"summary_arcface{base}.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Total frames: {total_frames}\n\n")
        f.write(f"Process time: {process_time:.2f} ms\n")

        f.write(f"Average TPF : {avg_tpf:.2f} ms/frame\n")
        f.write(f"Average FPS : {avg_fps:.2f}\n")
        f.write(f"Average CPU / core : {avg_cpu:.2f} %\n")
        f.write(f"Average GPU Memory : {avg_gpu_memory:.2f} MB\n")
        f.write(f"Average RAM : {avg_ram:.2f} MB\n")
        f.write(f"Detected frames: {detectedframe}\n")
        f.write(f"Detected unknown frames: {detectedunknownframe}\n")
        f.write(f"Detected rate: {detected_rate:.2f} %\n")
        f.write(f"True detection rate: {true_detection_rate:.2f} %\n")
        f.write(f"False negative rate: {false_negative_rate:.2f} %\n")

    print("ログを保存しました:", txt_path)

    return {
        "total_frames": total_frames,
        "process_time": process_time,
        "avg_tpf": avg_tpf,
        "avg_fps": avg_fps,
        "avg_cpu": avg_cpu,
        "avg_ram": avg_ram,
        "detected_frames": detectedframe,
        "detected_unknown_frames": detectedunknownframe,
        "detected_rate": detected_rate,
        "true_detection_rate": true_detection_rate,
        "false_negative_rate": false_negative_rate,
        "gpu_memory": avg_gpu_memory
    }

def avg(lst):
    return sum(lst) / len(lst)


# メイン処理開始
arcface = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
arcface.prepare(ctx_id=0, det_size=(640, 640))

print("ArcFace providers:",arcface.det_model.session.get_providers())

# 指定ディレクトリ構造
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# |-train(必須)
#   |-(名前)(一つ以上必要)
#     |-(画像)
#     |-(画像)
#   |-(名前)
#     |-(画像)
#
# |-videos(必須)
#   |-(動画ファイル)(一つ以上必要)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# !!!画像には顔を一つだけ写してください!!!
#
# ファイル名が日本語やスペースを含む場合、正しく動作しない可能性があります。

# 学習データの読み込み
if os.path.exists("arcface_encodings_avg.pkl"):
    user_input = input("学習データを再利用しますか？ (y/n): ")
else:
    print("学習データが存在しません。新規に学習データを作成します。")
    user_input = "n"
if user_input.lower() == "y":
    with open("arcface_encodings_avg.pkl", "rb") as f:
        avg_encodings = pickle.load(f)
    print("学習データの読み込み完了")
    print(avg_encodings);
    second_user_input = input("学習データを再利用しますか？ (y/n): ")

else:
    known_encodings = []
    known_names = []

    for person in os.listdir("train"):
        person_dir = os.path.join("train", person)
        if not os.path.isdir(person_dir):
            continue

        for file in os.listdir(person_dir):
            path = cv2.imread(os.path.join(person_dir, file))
            img = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
            faces = arcface.get(img)
            if len(faces) == 0:
                print(f"警告: {file} から顔を検出できませんでした")
                continue
            embedding = faces[0].normed_embedding
            known_encodings.append(embedding)
            known_names.append(person)

    person_embs = defaultdict(list)

    for emb, name in zip(known_encodings, known_names):
        person_embs[name].append(emb)

    avg_encodings = {}
    for name, embs in person_embs.items():
        # L2正規化平均embedding - 各次元ごとに平均を計算し、再度L2正規化
        m = np.mean(embs, axis=0)
        avg_encodings[name] = m / np.linalg.norm(m)

    print("学習データのエンコード完了")

    # 学習データの保存
    with open("arcface_encodings_avg.pkl", "wb") as f:
        pickle.dump(avg_encodings, f)

all_total_frames = []
all_avg_tpf = []
all_avg_fps = []
all_avg_cpu = []
all_avg_ram = []
all_detectedframe = []
all_detectedunknownframe = []
all_gpu_memory = []

video_dir = "videos"

for file in os.listdir(video_dir):
    if not file.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    video_path = os.path.join(video_dir, file)
    print("\n処理開始:", video_path)
    summary = process(video_path)

    all_total_frames.append(summary["total_frames"])
    all_avg_tpf.append(summary["avg_tpf"])
    all_avg_fps.append(summary["avg_fps"])
    all_avg_cpu.append(summary["avg_cpu"])
    all_avg_ram.append(summary["avg_ram"])
    all_detectedframe.append(summary["detected_frames"])
    all_detectedunknownframe.append(summary["detected_unknown_frames"])
    all_gpu_memory.append(summary["gpu_memory"])

txt_path = f"summary_all.txt"
j = 1
while os.path.exists(txt_path):
    txt_path = f"summary_all({j}).txt"
    j += 1

all_false_negative_rate = (sum(all_detectedunknownframe) / (sum(all_detectedframe) + sum(all_detectedunknownframe)))*100 if (sum(all_detectedframe) + sum(all_detectedunknownframe)) > 0 else 0

with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Total frames: {sum(all_total_frames)}\n\n")

        f.write(f"Average TPF : {avg(all_avg_tpf):.2f} ms/frame\n")
        f.write(f"Average FPS : {avg(all_avg_fps):.2f}\n")
        f.write(f"Average CPU / core : {avg(all_avg_cpu):.2f} %\n")
        f.write(f"Average GPU Memory : {avg(all_gpu_memory):.2f} MB\n")
        f.write(f"Average RAM : {avg(all_avg_ram):.2f} MB\n")
        f.write(f"Detected frames: {sum(all_detectedframe)}\n")
        f.write(f"Detected unknown frames: {sum(all_detectedunknownframe)}\n")
        f.write(f"Detected rate: {(sum(all_detectedframe) + sum(all_detectedunknownframe)) / sum(all_total_frames) * 100:.2f} %\n")
        f.write(f"True detection rate: {(sum(all_detectedframe) / sum(all_total_frames) * 100):.2f} %\n")
        f.write(f"False negative rate: {all_false_negative_rate} %\n")

print("ログを保存しました:", txt_path)