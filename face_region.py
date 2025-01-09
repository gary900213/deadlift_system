import cv2
from ultralytics import YOLO

def process_video(input_video_path, output_video_path, model_path):
    try:
        # 加載 YOLO 模型
        model = YOLO(model_path)

        # 打開輸入影片
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("無法打開影片！")5151515151515
            return

        # 取得影片資訊
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # 初始化輸出影片
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # 逐幀處理影片
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 使用 YOLOv8 模型檢測
            results = model(frame, conf=0.5)  # 設置信心閾值

            # 繪製檢測結果並進行模糊處理
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])  # 提取框的座標
                confidence = result.conf[0]  # 取得信心分數
                label = "Face"  # 標籤（可以改成其他名字）

                # 模糊檢測到的人臉區域
                face_region = frame[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y1:y2, x1:x2] = blurred_face  # 用模糊處理過的區域替換原圖

                # 繪製邊框和標籤
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 將處理後的幀寫入輸出影片
            out.write(frame)

        # 釋放資源
        cap.release()
        out.release()
        print(f"處理完成！輸出影片已儲存至 {output_video_path}")
    except Exception as e:
        print(f"處理過程中出現錯誤：{e}")




# 使用範例
input_video = "D:\\labdata\\MOCAP\\recordings_copy\\受試者66\\recording_20241227_100336\\vision1.avi"  # 輸入影片檔案路徑
output_video = "D:\\labdata\\MOCAP\\recordings_copy\\受試者66\\recording_20241227_100336\\vision1_1.avi"  # 輸出影片檔案路徑
model_weights = "yolov11n-face.pt"  # YOLOv8 人臉模型權重路徑
process_video(input_video, output_video, model_weights)





