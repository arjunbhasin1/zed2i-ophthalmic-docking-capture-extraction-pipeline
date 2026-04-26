import argparse
import cv2
import os
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="高精度测试 YOLO 模型检测针尖与目标点")

    # 默认使用你提供的视频和模型路径（请根据实际情况修改）
    default_source = r"C:\Users\Haiyang.Bian\PyCharmMiscProject\project_identify_needle_and_trocar\testmodel.mp4"
    default_model = r"C:\Users\Haiyang.Bian\PyCharmMiscProject\project_identify_needle_and_trocar\robot_project\models\best.pt"

    parser.add_argument("--source", type=str, default=default_source,
                        help="输入源：摄像头ID（如0）或视频文件路径")
    parser.add_argument("--model", type=str, default=default_model,
                        help="模型路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值，降低可提高召回率（推荐0.25）")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS的IoU阈值，降低可保留更多重叠框（推荐0.5）")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="推理图像尺寸，越大越精细但速度慢（推荐1280）")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="启用测试时增强(TTA)，提高精度但降低速度")
    parser.add_argument("--delay", type=int, default=30,
                        help="帧间延迟(毫秒)，值越大播放越慢")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="显示窗口缩放比例")
    parser.add_argument("--output", type=str, default="detection_result.mp4",
                        help="输出视频文件路径，默认保存为 detection_result.mp4")
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误：模型文件不存在 -> {args.model}")
        return

    # 检查视频源是否存在（若不是摄像头）
    if not args.source.isdigit() and not os.path.exists(args.source):
        print(f"错误：视频文件不存在 -> {args.source}")
        return

    print(f"加载模型: {args.model}")
    model = YOLO(args.model)
    print(f"模型加载成功，类别: {model.names}")

    # 打开视频源
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        source_name = f"Camera {args.source}"
    else:
        cap = cv2.VideoCapture(args.source)
        source_name = args.source

    if not cap.isOpened():
        print(f"无法打开源: {args.source}")
        return

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"原始视频: {width}x{height}, {fps:.2f} fps")

    # 视频写入器
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"输出视频将保存至: {args.output}")

    print(f"正在处理: {source_name}，按 ESC 键退出")

    # 可选：图像预处理增强（锐化），如需启用请取消注释
    # kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频处理完毕")
            break

        # 可选：图像锐化
        # frame = cv2.filter2D(frame, -1, kernel_sharpen)

        # 推理（启用 TTA、调整 iou 和置信度）
        results = model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, augment=args.augment)[0]

        # 可视化
        annotated = results.plot()

        # 提取针尖和目标点中心坐标并绘制
        needle_center = None
        target_center = None
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label == "needle":
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    needle_center = (cx, cy)
                    cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
                elif label == "target":
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    target_center = (cx, cy)
                    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        # 叠加坐标文本
        if needle_center:
            cv2.putText(annotated, f"Needle: {needle_center}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if target_center:
            cv2.putText(annotated, f"Target: {target_center}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示缩放
        display_frame = annotated
        if args.scale != 1.0:
            h, w = display_frame.shape[:2]
            new_w, new_h = int(w * args.scale), int(h * args.scale)
            display_frame = cv2.resize(display_frame, (new_w, new_h))

        cv2.imshow("High Precision Detection", display_frame)

        # 保存原始尺寸的结果（未缩放）
        if out is not None:
            out.write(annotated)

        # 按 ESC 退出
        if cv2.waitKey(args.delay) & 0xFF == 27:
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("程序结束")


if __name__ == "__main__":
    main()