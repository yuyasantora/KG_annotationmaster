import cv2
import numpy as np
import onnxruntime

# --- YOLOXのユーティリティ関数をインポート ---
# 前提：このファイルと utils.py が同じディレクトリにあること
try:
    from .utils import preproc as preprocess
    from .utils import multiclass_nms, demo_postprocess
    from .utils import COCO_CLASSES as DEFAULT_YOLOX_CLASSES
    from .utils import vis # ★デバッグ用にvisをここにインポート
except ImportError:
    # エラーメッセージをより分かりやすくする
    print("エラー: YOLOXのユーティリティファイルが見つかりません。")
    print("プロジェクト内に 'yolox' ディレクトリと、")
    print("その中に 'utils.py' 及び '__init__.py' があるか確認してください。")
    raise

class YOLOXONNXPredictor:
    """
    YOLOXのONNXモデルを使用して物体検出を行うクラス。
    """
    def __init__(self, model_bytes, input_shape_str="640,640", class_names=None):
        """
        モデルのバイトデータと設定で初期化します。

        Args:
            model_bytes (bytes): ONNXモデルファイルのバイトデータ。
            input_shape_str (str): "高さ,幅" の形式の文字列 (例: "640,640")。
            class_names (list or tuple, optional): 
                物体のクラス名のリスト。指定されない場合、
                YOLOXのデフォルトCOCOクラス名を使用します。
        """
        self.input_shape = tuple(map(int, input_shape_str.split(',')))
        
        # クラス名の設定
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = DEFAULT_YOLOX_CLASSES

        # ONNXランタイムセッションの初期化
        try:
            self.session = onnxruntime.InferenceSession(model_bytes)
        except Exception as e:
            # アプリケーション側で捕捉できるようにRuntimeErrorを送出
            raise RuntimeError(f"ONNXモデルのロードに失敗しました: {e}")

        # モデルの入力名を取得
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, origin_img_bgr, score_thr=0.3, nms_thr=0.45):
        """
        単一の画像 (OpenCV BGR形式) に対して物体検出を実行します。
        この実装は、YOLOX公式の onnx_inference.py を参考にしています。
        パディングを考慮した座標のスケールバックを行います。
        """
        if origin_img_bgr is None:
            return []

        # 1. 画像の前処理
        # preprocess (preproc) は、リサイズ・パディング後の画像と、
        # 元画像への変換比率(ratio)を返す。
        img_processed, ratio = preprocess(origin_img_bgr, self.input_shape)

        # 2. 推論の実行
        # バッチサイズ1の想定
        ort_inputs = {self.input_name: img_processed[None, :, :, :]}
        output = self.session.run(None, ort_inputs)
        
        # 3. 推論結果の後処理 (デコード)
        # YOLOXでは、p6=False が yolox-s,m,l,x に対応
        # demo_postprocess は、モデルの出力(グリッド座標)を
        # 入力画像サイズ(例: 416x416)のスケールでの (cx, cy, w, h) に変換する。
        predictions = demo_postprocess(output[0], self.input_shape, p6=False)[0]

        if predictions is None or predictions.shape[0] == 0:
            return []

        # 4. NMS (Non-Maximum Suppression) の実行
        
        # ボックス座標の準備 (cx, cy, w, h)
        boxes = predictions[:, :4]
        # スコアの準備 (オブジェクトネススコア * クラススコア)
        scores = predictions[:, 4:5] * predictions[:, 5:]

        # (cx, cy, w, h) から (x1, y1, x2, y2) へ変換
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        
        # ★★★ここが最も重要な座標変換部分★★★
        # パディングを考慮して、元の画像サイズにスケールバックする
        # preproc関数で追加されたパディング量(上下左右の余白)を考慮せずに
        # 単純に ratio で割ると座標がずれるため、
        # YOLOXのデモコードでは NMS の後に ratio で割るのではなく、
        # NMS の前に ratio で割ることで、パディングの影響を受けたままの座標で
        # NMS を行い、その後にパディング分を補正する、というような実装は見られない。
        # 代わりに、demo_postprocess がパディングを考慮しているか、
        # もしくは preproc が返す ratio がパディング込みの変換率である必要がある。
        # YOLOXの標準的な preproc は、パディングを除去するような ratio を返す。
        boxes_xyxy /= ratio
        
        # multiclass_nms を適用して重複するボックスを削除
        # dets の形式: [x1, y1, x2, y2, score, class_idx]
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)

        # --- ★★★ ここからデバッグコードを追加 ★★★ ---
        if dets is not None:
            final_boxes, final_scores, final_cls_ids = dets[:, :4], dets[:, 4], dets[:, 5]
            
            # 元の画像(origin_img_bgr)のコピーに描画
            result_img = vis(origin_img_bgr.copy(), final_boxes, final_scores, final_cls_ids,
                             conf=score_thr, class_names=self.class_names)
            
            # デバッグ用の画像として保存 (ファイル名は固定にする)
            debug_output_path = f"debug_annotation_result.jpg"
            cv2.imwrite(debug_output_path, result_img)
            print(f"★★★ デバッグ用画像を出力しました: {debug_output_path} ★★★")
        # --- ★★★ デバッグコードここまで ★★★ ---

        detected_objects = []
        if dets is not None:
            # vis関数でのデバッグ描画用
            # final_boxes, final_scores, final_cls_ids = dets[:, :4], dets[:, 4], dets[:, 5]
            # debug_img = vis(origin_img_bgr.copy(), final_boxes, final_scores, final_cls_ids, conf=score_thr, class_names=self.class_names)
            # cv2.imwrite(f"debug_predict_output.jpg", debug_img)

            for i in range(dets.shape[0]):
                box = dets[i, :4]
                # 座標が画像の範囲外に出ないようにクリッピングする
                box[0] = max(0, box[0])
                box[1] = max(0, box[1])
                box[2] = min(origin_img_bgr.shape[1], box[2]) # width
                box[3] = min(origin_img_bgr.shape[0], box[3]) # height
                
                score = float(dets[i, 4])
                cls_idx = int(dets[i, 5])
                
                # クラスIDをクラス名に変換
                class_name = self.class_names[cls_idx] if 0 <= cls_idx < len(self.class_names) else "unknown"

                detected_objects.append({
                    "label_name": class_name,
                    "xmin": int(box[0]),
                    "ymin": int(box[1]),
                    "xmax": int(box[2]),
                    "ymax": int(box[3]),
                    "score": score
                })
                
        return detected_objects

# --- スクリプトとして直接実行した場合のテスト用コード (任意) ---
if __name__ == '__main__':
    # このスクリプトを直接実行した際のテスト用コード
    # 使い方: python yolox_onnx_predictor.py <onnx_model_path> <image_path>
    import sys
    
    if len(sys.argv) != 3:
        print("使い方: python yolox_onnx_predictor.py <onnx_model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    try:
        # 1. モデルの初期化
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        
        # COCOクラス名で初期化
        predictor = YOLOXONNXPredictor(model_bytes, class_names=DEFAULT_YOLOX_CLASSES)
        print(f"モデル '{model_path}' をロードしました。")

        # 2. 画像の読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"エラー: 画像ファイル '{image_path}' を読み込めません。")
            sys.exit(1)
        print(f"画像 '{image_path}' を読み込みました。")

        # 3. 推論の実行
        print("推論を実行中...")
        detections = predictor.predict(image, score_thr=0.4)

        # 4. 結果の表示
        if not detections:
            print("物体は検出されませんでした。")
        else:
            print(f"{len(detections)} 個の物体を検出しました:")
            for det in detections:
                print(f"  - ラベル: {det['label_name']}, "
                      f"信頼度: {det['score']:.2f}, "
                      f"BBox: [{det['xmin']}, {det['ymin']}, {det['xmax']}, {det['ymax']}]")
            
            # 結果を描画して表示 (デバッグ用)
            result_image = vis(image.copy(), 
                               [np.array([d['xmin'], d['ymin'], d['xmax'], d['ymax']]) for d in detections],
                               [d['score'] for d in detections],
                               [predictor.class_names.index(d['label_name']) for d in detections],
                               conf=0.4,
                               class_names=predictor.class_names)
            
            cv2.imshow("Detection Result", result_image)
            print("結果画像が表示されました。任意のキーを押して終了します。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"エラーが発生しました: {e}")
