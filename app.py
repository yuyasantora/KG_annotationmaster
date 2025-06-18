import streamlit as st
import openai
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from search import create_index_faiss, find_closest_vectors_faiss
from openai import OpenAI
import os
import subprocess
import io
import traceback
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
import xml.etree.ElementTree as ET
import sys
from datetime import datetime, timedelta
import cv2
from yolox.yolox_onnx_predictor import YOLOXONNXPredictor
import json
import re
from dotenv import load_dotenv
import boto3
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy import func, text, or_
from sqlalchemy.dialects.postgresql import JSONB


load_dotenv()

DB_PATH = "annotation_data.db"

# S3クライアントの初期化
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_S3_REGION")
)
# --- ★ 設定ここまで ★ ---

TEMP_SESSIONS_BASE_DIR = "temp_sessions"
OLD_SESSION_MAX_AGE_HOURS = 24 # 起動時クリーンアップ対象（例: 24時間）
CURRENT_SESSION_TIMEOUT_SECONDS = 3600 # 現在のセッションのタイムアウト（例: 1時間 = 3600秒）

# 事前定義モデル
PRESET_MODELS = {
    "信号検出":{
        "path":"yolox/onnx_models/tlr_car_ped_yolox_s_batch_1.onnx", # バッチサイズ1のモデルを想定
        "input_shape":"416,416",
        "class_names":[ "pedestrian_traffic_light", "traffic_light"]
    },
}

# --- SQLAlchemyモデル定義 ---

# モデルクラスが継承するためのBaseクラスを作成
Base = declarative_base()

# 画像情報を格納するテーブルモデル
class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    s3_key = Column(String, unique=True)
    label = Column(String, nullable=True)
    annotations = Column(JSON, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

# --- データベース接続設定 ---
# 元のSQLite接続設定
# DATABASE_FILE = "annotation.db"
# engine = create_engine(f"sqlite:///{DATABASE_FILE}")

# 新しいPostgreSQLへの接続設定
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    st.error("DATABASE_URLが設定されていません。.envファイルを確認してください。")
    st.stop()

engine = create_engine(DATABASE_URL)

# --- テーブル作成 ---
# データベースにテーブルが存在しない場合、テーブルを初回作成する
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def parse_pascal_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({'label_name': label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
    return objects

def get_classes_from_xml_dir(xml_dir_path):
    """指定されたディレクトリ内のすべてのPASCAL VOC XMLからクラス名を抽出する"""
    annotated_classes = set()
    if not os.path.isdir(xml_dir_path):
        return annotated_classes
    
    for filename in os.listdir(xml_dir_path):
        if filename.endswith(".xml"):
            xml_file_path = os.path.join(xml_dir_path, filename)
            try:
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name_element = obj.find("name")
                    if name_element is not None and name_element.text:
                        annotated_classes.add(name_element.text.strip())
            except ET.ParseError:
                # 不正なXMLはスキップ
                continue
    return annotated_classes

def format_caption(rank, description):
    return f"**Rank {rank}:**\n{description}"

@st.cache_data
def search_images_in_db(search_term):
    """データベースで画像を検索し、結果を返す。結果はキャッシュされる。"""
    db = SessionLocal()
    try:
        query = db.query(Image)
        if search_term:
            search_pattern = f'%{search_term}%'
            from sqlalchemy import or_
            query = query.filter(
                or_(
                    Image.filename.ilike(search_pattern),
                    Image.label.ilike(search_pattern)
                )
            )
        return query.order_by(Image.id.desc()).all()
    finally:
        db.close()

def run_image_search_app():
    """データベースに登録された画像の一覧表示と検索を行うページ"""
    st.header("登録済み画像の検索・閲覧")

    # --- 検索UI ---
    search_term = st.text_input("ファイル名や分類ラベルで検索:")

    # --- キャッシュされた関数を呼び出して画像情報を取得 ---
    all_images = search_images_in_db(search_term)

    if not all_images:
        st.info("データベースに登録されている画像はありません。")
        return
        
    st.write(f"**{len(all_images)}** 件の画像が見つかりました。")
    st.markdown("---")

    # --- 画像一覧の表示 ---
    if 'temp_dir_session' not in st.session_state or not os.path.exists(st.session_state.temp_dir_session):
        st.session_state.temp_dir_session = setup_session_temp_dir()
    temp_dir = st.session_state.temp_dir_session

    cols = st.columns(4) # 4列で表示
    for i, img_obj in enumerate(all_images):
        with cols[i % 4]:
            image_bytes = get_image_bytes_from_s3(img_obj.s3_key)
            
            if image_bytes:
                st.image(image_bytes, use_column_width=True)
                st.markdown(f"**{img_obj.filename}**")
                # DBから取得した最新の分類ラベルを表示
                st.caption(f"分類: {img_obj.label or '(未設定)'}")
                anno_count = len(img_obj.annotations) if img_obj.annotations else 0
                st.caption(f"物体数: {anno_count}")
            else:
                st.warning(f"表示不可:\n{img_obj.filename}")
                                    
def setup_session_temp_dir():
    base_temp_path = Path(TEMP_SESSIONS_BASE_DIR)
    base_temp_path.mkdir(exist_ok=True)
    session_dir_name = f"session_{str(uuid.uuid4())[:8]}"
    session_temp_path = base_temp_path / session_dir_name
    session_temp_path.mkdir(exist_ok=True, parents=True)
    
    if 'last_activity_time' in st.session_state:
        st.session_state.last_activity_time = time.time()
    else:
        st.session_state.last_activity_time = time.time()
    return str(session_temp_path)

def cleanup_session_temp_dir(session_dir_path):
    if session_dir_path and os.path.exists(session_dir_path):
        try:
            shutil.rmtree(session_dir_path)
        except Exception as e:
            st.error(f"一時ディレクトリの削除に失敗しました: {session_dir_path}, Error: {e}")

def generate_pascal_voc_xml(images_to_export: list[Image], tmpdir_path: Path):
    """
    検出結果からPASCAL VOC XML文字列を生成する。
    objects: [{'label_name': str, 'xmin': int, ...}, ...]
    """
    missing_files = []

    image_dir = tmpdir_path / "images"
    annotations_dir = tmpdir_path / "annotations"

    for image_obj in images_to_export:
        local_image_path = get_image_from_s3(image_obj.s3_key, str(tmpdir_path))
        if not local_image_path:
            missing_files.append(image_obj.filename)
            continue

        try:
            with PILImage.open(local_image_path) as img:
                width, height = img.size
        except Exception as e:
            st.warning(f"画像ファイル{image_obj.filename}を読み込めませんでした。{e}")
            missing_files.append(image_obj.filename)
            continue

        shutil.copy(local_image_path, image_dir / image_obj.filename)

        detected_objects = image_obj.annotations if image_obj.annotations else []
        xml_content = generate_pascal_voc_xml_content(image_obj.filename, local_image_path, width, height, detected_objects)
        xml_filename = Path(image_obj.filename).stem + ".xml"
        with open(annotations_dir / xml_filename, "w", encoding="utf-8") as f:
            f.write(xml_content)

    return len(images_to_export) - len(missing_files), missing_files

def generate_coco_dataset(images_to_export: list[Image], tmpdir_path: Path):
    """
    検出結果からCOCO JSON文字列を生成する。"""
    
    # 全アノテーションからカテゴリを収集
    all_labels = sorted(list(set(
        obj["label_name"] for img in images_to_export if img.annotations for obj in img.annotations

    )))
    categories = [{"id": i+1, "name": label, "supercategory": "none"} for i, label in enumerate(all_labels)]
    category_map = {cat["name"]: cat["id"] for cat in categories}

    coco_output = {
        "info": {"description": "Generated by Annotation Tool", "version": "1.0", "year": datetime.now().year, "date_created": datetime.now().strftime('%Y/%m/%d')},
        "licenses": [], "images": [], "annotations": [], "categories": categories
    }

    annotation_id_counter = 1
    missing_files = []
    images_dir = tmpdir_path / "images"

    for image_obj in images_to_export:
        local_image_path = get_image_from_s3(image_obj.s3_key, str(tmpdir_path))
        if not local_image_path:
            missing_files.append(image_obj.filename)
            continue
            
        try:
            with PILImage.open(local_image_path) as img:
                width, height = img.size
        except Exception as e:
            st.warning(f"画像ファイル {image_obj.filename} を開けません: {e}")
            missing_files.append(image_obj.filename)
            continue

        shutil.copy(local_image_path, images_dir / image_obj.filename)
        
        image_coco_id = len(coco_output["images"]) + 1
        coco_output["images"].append({"id": image_coco_id, "file_name": image_obj.filename, "width": width, "height": height})
        
        if image_obj.annotations:
            for anno in image_obj.annotations:
                xmin, ymin, xmax, ymax = anno['xmin'], anno['ymin'], anno['xmax'], anno['ymax']
                bbox_width, bbox_height = xmax - xmin, ymax - ymin
                if anno['label_name'] in category_map:
                    coco_output["annotations"].append({
                        "id": annotation_id_counter, "image_id": image_coco_id, 
                        "category_id": category_map[anno['label_name']], 
                        "bbox": [xmin, ymin, bbox_width, bbox_height], 
                        "area": bbox_width * bbox_height, "iscrowd": 0, "segmentation": []
                    })
                    annotation_id_counter += 1
    
    with open(tmpdir_path / "annotations" / "instances.json", "w") as f:
        json.dump(coco_output, f, indent=4)
        
    return len(images_to_export) - len(missing_files), missing_files

def generate_yolo_dataset(images_to_export: list[Image], tmpdir_path: Path):
    """YOLO Darknet形式のデータセットを生成し、統計情報を返す"""
    st.info("YOLO Darknet形式のデータセットを生成しています...")
    
    all_labels = sorted(list(set(
        obj['label_name'] 
        for img in images_to_export if img.annotations 
        for obj in img.annotations
    )))
    class_map = {label: i for i, label in enumerate(all_labels)}
    with open(tmpdir_path / "classes.txt", "w") as f:
        for label in all_labels: f.write(f"{label}\n")

    missing_files = []
    images_dir = tmpdir_path / "images"
    annotations_dir = tmpdir_path / "annotations"

    for image_obj in images_to_export:
        local_image_path = get_image_from_s3(image_obj.s3_key, str(tmpdir_path))
        if not local_image_path:
            missing_files.append(image_obj.filename)
            continue

        try:
            with PILImage.open(local_image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            st.warning(f"画像ファイル {image_obj.filename} を開けません: {e}")
            missing_files.append(image_obj.filename)
            continue
            
        shutil.copy(local_image_path, images_dir / image_obj.filename)
        
        yolo_lines = []
        if image_obj.annotations:
            for anno in image_obj.annotations:
                class_id = class_map[anno['label_name']]
                xmin, ymin, xmax, ymax = anno['xmin'], anno['ymin'], anno['xmax'], anno['ymax']
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        txt_filename = Path(image_obj.filename).stem + ".txt"
        with open(annotations_dir / txt_filename, "w") as f:
            f.write("\n".join(yolo_lines))
            
    return len(images_to_export) - len(missing_files), missing_files

def get_image_from_s3(s3_key, temp_dir):
    """S3から画像をダウンロードし、一時ファイルのパスを返す"""
    if not s3_key:
        return None
    try:
        local_path = os.path.join(temp_dir, os.path.basename(s3_key))
        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        return local_path
    except Exception as e:
        st.warning(f"S3からの画像取得に失敗: {s3_key}, Error: {e}")
        return None

@st.cache_data
def get_image_bytes_from_s3(s3_key):
    """S3から画像をダウンロードし、そのバイトデータを返す。結果はキャッシュされる。"""
    if not s3_key:
        return None
    try:
        s3_object = s3.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return s3_object['Body'].read()
    except Exception as e:
        st.warning(f"S3からの画像取得に失敗: {s3_key}, Error: {e}")
        return None

@st.cache_data
def get_bytes_from_uploaded_file(uploaded_file):
    """アップロードされたファイルオブジェクトからバイトデータを返す。結果はキャッシュされる。"""
    return uploaded_file.getvalue()

def generate_pascal_voc_xml_content(filename, path, width, height, objects):
    """
    検出結果からPASCAL VOC XMLの文字列を生成する。
    objects: [{'label_name': str, 'xmin': int, ...}, ...]
    """
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = "images"
    ET.SubElement(annotation, 'filename').text = filename
    ET.SubElement(annotation, 'path').text = str(path)
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    ET.SubElement(annotation, 'segmented').text = '0'

    # objectsがNoneや空リストの場合でもエラーにならないようにチェック
    if objects:
        for obj_dict in objects:
            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = obj_dict['label_name']
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(obj_dict['xmin'])
            ET.SubElement(bndbox, 'ymin').text = str(obj_dict['ymin'])
            ET.SubElement(bndbox, 'xmax').text = str(obj_dict['xmax'])
            ET.SubElement(bndbox, 'ymax').text = str(obj_dict['ymax'])

    # ElementTreeから整形されたXML文字列を生成
    ET.indent(annotation, space="  ")
    xml_string = ET.tostring(annotation, encoding='unicode')
    
    return f'<?xml version="1.0" ?>\n{xml_string}'

def convert_image_to_vector():
    # --- セッション管理変数の初期化 ---
    if 'staged_files' not in st.session_state:
        st.session_state.staged_files = {} # filename: UploadedFile
    if 'annotation_data' not in st.session_state:
        st.session_state.annotation_data = {} # filename: {annotations: [], label: ""}

    # --- 一時ディレクトリの準備 ---
    if 'temp_dir_session' not in st.session_state or not os.path.exists(st.session_state.temp_dir_session):
        st.session_state.temp_dir_session = setup_session_temp_dir()
    temp_dir = st.session_state.temp_dir_session
    local_image_storage_dir = os.path.join(temp_dir, "images_for_labeling")
    open_labeling_output_dir_base = os.path.join(temp_dir, "annotations_output")
    open_labeling_pascal_voc_output_dir = os.path.join(open_labeling_output_dir_base, "PASCAL_VOC")
    os.makedirs(local_image_storage_dir, exist_ok=True)
    os.makedirs(open_labeling_pascal_voc_output_dir, exist_ok=True) 

    # --- 1. ファイルアップロード ---
    st.header("1. 画像をアップロード")
    uploaded_files = st.file_uploader("画像ファイルを選択", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="file_uploader_main")
    if uploaded_files:
        new_files_added = False
        for f in uploaded_files:
            if f.name not in st.session_state.staged_files:
                st.session_state.staged_files[f.name] = f
                # 対応するアノテーションデータも初期化
                st.session_state.annotation_data[f.name] = {"annotations": [], "label": ""}
                new_files_added = True
        if new_files_added:
            st.rerun()

    if not st.session_state.staged_files:
        st.info("画像をアップロードすると、ここにプレビューと処理オプションが表示されます。")
        st.stop()

    # --- アップロードされた画像のプレビュー ---
    st.subheader(f"{len(st.session_state.staged_files)} 件の画像がステージング中")
    with st.expander("画像プレビューを開く/閉じる"):
        cols = st.columns(4)
        for i, (filename, file_data) in enumerate(st.session_state.staged_files.items()):
            with cols[i % 4]:
                image_bytes = get_bytes_from_uploaded_file(file_data)
                st.image(image_bytes, caption=filename, use_column_width=True)
    
    st.markdown("---")

    # --- 2. 自動アノテーション ---
    st.header("2. 自動アノテーション (任意)")
    auto_annotate_method = st.radio("方法を選択", ("プリセットモデルを使用", "カスタムモデルをアップロード"), key="auto_annotate_method_radio")

    onnx_model_bytes, class_names, input_shape = None, None, "640,640"
    if auto_annotate_method == "プリセットモデルを使用" and PRESET_MODELS:
        model_name = st.selectbox("モデルを選択", options=list(PRESET_MODELS.keys()))
        model_info = PRESET_MODELS[model_name]
                    if os.path.exists(model_info["path"]):
                        with open(model_info["path"], "rb") as f:
                onnx_model_bytes = f.read()
            class_names = model_info.get("class_names")
            input_shape = model_info.get("input_shape", "640,640")
                    else:
            st.error(f"モデルファイルが見つかりません: {model_info['path']}")
    elif auto_annotate_method == "カスタムモデルをアップロード":
        f = st.file_uploader("ONNXモデル (.onnx)", type=["onnx"])
        if f: onnx_model_bytes = f.read()

    confidence_threshold = st.slider("信頼度のしきい値", 0.0, 1.0, 0.3, 0.05)
    
    if st.button("自動アノテーションを実行"):
        if onnx_model_bytes:
            try:
                predictor = YOLOXONNXPredictor(model_bytes=onnx_model_bytes, input_shape_str=input_shape, class_names=class_names)
                with st.spinner("自動アノテーションを実行中..."):
                    for filename, file_data in st.session_state.staged_files.items():
                        image_bytes = get_bytes_from_uploaded_file(file_data)
                        np_array = np.frombuffer(image_bytes, np.uint8)
                        cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                        detected = predictor.predict(cv_image, score_thr=confidence_threshold)
                        # 結果をXMLとして保存
                        pil_img = PILImage.open(io.BytesIO(image_bytes))
                        xml_content = generate_pascal_voc_xml_content(filename, "s3_path_placeholder", pil_img.width, pil_img.height, detected)
                        with open(os.path.join(open_labeling_pascal_voc_output_dir, f"{Path(filename).stem}.xml"), "w") as f:
                            f.write(xml_content)
                st.success("自動アノテーションが完了しました。下記OpenLabelingで結果を確認・修正できます。")
            except Exception as e:
                st.error(f"自動アノテーション中にエラー: {e}")
        else:
            st.warning("モデルが選択またはアップロードされていません。")
    
    st.markdown("---")
    
    # --- 3. 手動アノテーション ---
    st.header("3. 手動アノテーション (OpenLabeling)")
    st.info("自動アノテーションの結果や、新規のアノテーションをこのツールで編集します。")
    
    # クラスリストの準備
    class_list_file_path = os.path.abspath(os.path.join("OpenLabeling", "main", "class_list.txt"))
    default_classes_str = "object\nperson\nvehicle"
    try:
        if os.path.exists(class_list_file_path):
            with open(class_list_file_path, "r") as f: default_classes_str = f.read()
    except Exception: pass
    
    edited_classes_str = st.text_area("クラスリスト (1行1クラス)", value=default_classes_str, height=150)

    if st.button("OpenLabelingを起動"):
        # ステージング中の画像をOpenLabelingの入力フォルダに書き出す
        for filename, file_data in st.session_state.staged_files.items():
            with open(os.path.join(local_image_storage_dir, filename), "wb") as f:
                f.write(get_bytes_from_uploaded_file(file_data))

        # クラスリストをファイルに保存
        with open(class_list_file_path, "w") as f: f.write(edited_classes_str)

        command = [sys.executable, os.path.abspath("OpenLabeling/main/main.py"), "--input_dir", os.path.abspath(local_image_storage_dir), "--output_dir", os.path.abspath(open_labeling_output_dir_base), "--draw-from-PASCAL-files"]
        subprocess.Popen(command, cwd=os.path.abspath("OpenLabeling/main"))
        st.success("OpenLabelingが別ウィンドウで起動しました。")
        
        st.markdown("---")
    
    # --- 4. 分類ラベル入力と最終保存 ---
    st.header("4. 分類ラベル入力とDBへの保物体ラベルの集計中にエラー: (psycopg2.errors.UndefinedFunction) function jsonb_array_elements(json) does not exist LINE 2: FROM (SELECT images.id AS image_id, jsonb_array_elements(ima... ^ HINT: No function matches the given name and argument types. You might need to add explicit type casts.

[SQL: SELECT anon_1.label_name AS anon_1_label_name, count(distinct(anon_1.image_id)) AS count_1 FROM (SELECT images.id AS image_id, jsonb_array_elements(images.annotations) ->> %(jsonb_array_elements_1)s AS label_name FROM images WHERE images.annotations IS NOT NULL AND jsonb_array_length(images.annotations) > %(jsonb_array_length_1)s) AS anon_1 GROUP BY anon_1.label_name ORDER BY anon_1.label_name] [parameters: {'jsonb_array_elements_1': 'label_name', 'jsonb_array_length_1': 0}] (Background on this error at: https://sqlalche.me/e/20/f405)
                for filename, file_data in st.session_state.staged_files.items():
                    # (1) アノテーションXMLを読み込む
                    xml_path = os.path.join(open_labeling_pascal_voc_output_dir, f"{Path(filename).stem}.xml")
                    if os.path.exists(xml_path):
                        st.session_state.annotation_data[filename]["annotations"] = parse_pascal_voc_xml(xml_path)
                    
                    # (2) S3へアップロード
                    s3_key = f"images/{uuid.uuid4()}_{filename}"
                    file_data.seek(0)
                    s3.upload_fileobj(file_data, S3_BUCKET_NAME, s3_key, ExtraArgs={'ContentType': file_data.type})

                    # (3) DBへ保存 (SQLAlchemy使用)
                    pil_img = PILImage.open(file_data)
                    
                    # 存在確認 (Upsert)
                    image_obj = db.query(Image).filter(Image.s3_key == s3_key).first()
                    if image_obj: # 更新
                        image_obj.filename = filename
                        image_obj.width = pil_img.width
                        image_obj.height = pil_img.height
                        image_obj.label = st.session_state.annotation_data[filename]["label"] or None
                        image_obj.annotations = st.session_state.annotation_data[filename]["annotations"]
                    else: # 新規作成
                        image_obj = Image(
                            filename=filename,
                            s3_key=s3_key,
                            width=pil_img.width,
                            height=pil_img.height,
                            label=st.session_state.annotation_data[filename]["label"] or None,
                            annotations=st.session_state.annotation_data[filename]["annotations"]
                        )
                        db.add(image_obj)
                
                db.commit()
                st.success(f"{len(st.session_state.staged_files)}件の画像の処理が完了しました。")
                # 正常に終了したらセッションをクリア
                st.session_state.staged_files.clear()
                st.session_state.annotation_data.clear()
                st.rerun()

                except Exception as e:
                db.rollback()
                st.error(f"保存中にエラーが発生しました: {e}")
                st.error(traceback.format_exc())
            finally:
                db.close()

    if st.session_state.staged_files:
        if st.button("現在の処理を全てクリア"):
            st.session_state.staged_files.clear()
            st.session_state.annotation_data.clear()
        st.rerun()

def cleanup_old_session_dirs():
    base_path = Path(TEMP_SESSIONS_BASE_DIR)
    if not base_path.exists():
        return
    now = datetime.now()
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("session_"):
            try:
                dir_modified_time = datetime.fromtimestamp(item.stat().st_mtime)
                if now - dir_modified_time > timedelta(hours=OLD_SESSION_MAX_AGE_HOURS):
                    shutil.rmtree(item)
            except Exception:
                pass 

def check_session_timeout():
    if 'last_activity_time' not in st.session_state:
        st.session_state.last_activity_time = time.time()
        return False

    if time.time() - st.session_state.last_activity_time > CURRENT_SESSION_TIMEOUT_SECONDS:
        current_session_dir_name = "不明なセッション"
        if 'temp_dir_session' in st.session_state and st.session_state.temp_dir_session:
            current_session_dir_name = os.path.basename(st.session_state.temp_dir_session)
            cleanup_session_temp_dir(st.session_state.temp_dir_session)
        
        st.session_state.clear()
        st.session_state.temp_dir_session = setup_session_temp_dir()
        st.session_state.current_page = "画像を登録する"
        
        st.info(f"長時間操作がなかったため、セッションデータ ({current_session_dir_name}) をクリアし、新しいセッションを開始しました。")
        st.rerun()
        return True
    return False

def parse_label_from_option(option_str):
    """ "label (123枚)" という形式の文字列からラベル名だけを抽出する """
    match = re.match(r"^(.*) \(\d+枚\)$", option_str)
    if match:
        return match.group(1)
    return option_str

def create_dataset_page():
    """データセット作成ページのメイン関数。タブUIを使用。"""
    st.header("データセット作成")
    
    tab1, tab2 = st.tabs(["物体検出データセット", "画像分類データセット"])

    # --- 物体検出データセット用タブ ---
    with tab1:
        st.info("特定の物体ラベルが含まれる画像を抽出し、選択したフォーマットでデータセットを作成します。")
        
        db = SessionLocal()
        try:
            # アノテーションを持つ画像からラベル名と、そのラベルを持つ画像のユニークな数を集計
            subquery = db.query(
                Image.id.label("image_id"),
                func.jsonb_array_elements(Image.annotations).op('->>')('label_name').label("label_name")
            ).filter(Image.annotations != None, Image.annotations != text("'[]'")).subquery()
            query = db.query(
                subquery.c.label_name,
                func.count(func.distinct(subquery.c.image_id))
            ).group_by(subquery.c.label_name).order_by(subquery.c.label_name)
            available_labels = query.all()
        except Exception as e:
            st.error(f"物体ラベルの集計中にエラー: {e}")
            available_labels = []
        finally:
            db.close()

        if not available_labels:
            st.warning("データベースに物体のアノテーションデータがありません。")
        else:
            dataset_format = st.selectbox("フォーマット:", ("PASCAL VOC", "COCO", "YOLO Darknet"), key="detection_format_select")
            options_with_counts = [f"{label} ({count}枚)" for label, count in available_labels]
            selected_options = st.multiselect("含める物体ラベル (未選択=全て):", options_with_counts, key="detection_labels_multiselect")

            if st.button("検出データセットを生成", key="generate_detection_dataset_btn"):
                selected_labels_to_filter = [parse_label_from_option(opt) for opt in selected_options]
                
                db = SessionLocal()
                try:
                    query = db.query(Image).filter(Image.annotations != None, Image.annotations != text("'[]'"))
                    if selected_labels_to_filter:
                        # '?' 演算子はjsonb配列のトップレベルのキーの存在をチェックする
                        # 各オブジェクトの'label_name'キーをチェックするには@>演算子などがより適切
                        # ここでは ANY と ->> を使って配列内のオブジェクトのプロパティをチェック
                         query = query.filter(text("EXISTS (SELECT 1 FROM jsonb_array_elements(images.annotations) as elem WHERE elem->>'label_name' = ANY(:labels))").bindparams(labels=selected_labels_to_filter))

                    images_to_export = query.all()
                    
                    if not images_to_export:
                        st.warning("対象データが見つかりませんでした。"); return

                    with st.spinner(f"{len(images_to_export)}件の画像を処理中..."):
                        import tempfile; import zipfile
                        with tempfile.TemporaryDirectory() as tmpdir:
                            dataset_root = Path(tmpdir) / "dataset"
                            (dataset_root / "images").mkdir(parents=True)
                            (dataset_root / "annotations").mkdir()
                            
                            gen_func = {"PASCAL VOC": generate_pascal_voc_dataset, "COCO": generate_coco_dataset, "YOLO Darknet": generate_yolo_dataset}[dataset_format]
                            # ★★★ 修正ポイント ★★★
                            # cursorやidリストの代わりに、Imageオブジェクトのリストを直接渡す
                            generated_count, missing_files = gen_func(images_to_export, dataset_root)
                            
                            if generated_count == 0:
                                st.warning("有効なデータがなかったため、zipファイルを生成できませんでした。"); return
                            
                            zip_path_str = os.path.join(tmpdir, "export"); shutil.make_archive(zip_path_str, 'zip', dataset_root)
                            zip_filename = f"{dataset_format.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                            st.download_button(label=f"📥 {dataset_format} データセットをDL", data=Path(f"{zip_path_str}.zip").read_bytes(), file_name=zip_filename, mime="application/zip")
                            if missing_files: st.warning(f"{len(missing_files)}件のファイルが欠損または読み込みエラーでした: {', '.join(missing_files[:5])}...")
                except Exception as e:
                    st.error(f"データセット生成中にエラー: {e}")
                    st.error(traceback.format_exc())
                finally:
                    db.close()

    # --- 画像分類データセット用タブ ---
    with tab2:
        st.info("特定の分類ラベルが付与された画像を抽出し、フォルダ分けされたデータセットを作成します。")
        db = SessionLocal()
        try:
            query = db.query(Image.label, func.count(Image.id)).filter(Image.label != None, Image.label != '').group_by(Image.label).order_by(Image.label)
            available_labels = query.all()
        except Exception as e:
            st.error(f"分類ラベルの集計中にエラー: {e}"); available_labels = []
        finally:
            db.close()
        
        if not available_labels:
            st.warning("データベースに分類ラベル付きの画像データがありません。")
        else:
            options_with_counts = [f"{label} ({count}枚)" for label, count in available_labels]
            selected_options = st.multiselect("含める分類ラベル (未選択=全て):", options_with_counts, key="classification_labels_multiselect")
            
            if st.button("分類データセットを生成", key="generate_classification_dataset_btn"):
                labels_to_filter = [parse_label_from_option(opt) for opt in selected_options]
                if not labels_to_filter:
                    # 未選択の場合は全てのラベルを対象とする
                    labels_to_filter = [label for label, count in available_labels]

                db = SessionLocal()
                try:
                    images_to_export = db.query(Image).filter(Image.label.in_(labels_to_filter)).all()
                    if not images_to_export:
                        st.warning("対象データが見つかりませんでした。"); return

                    with st.spinner(f"処理中... {len(images_to_export)}件"):
                        import tempfile; import zipfile
                        with tempfile.TemporaryDirectory() as tmpdir:
                            dataset_root = Path(tmpdir) / "classification_dataset"
                            dataset_root.mkdir()
                            missing_files_count = 0
                            
                            for image_obj in images_to_export:
                                label_dir = dataset_root / image_obj.label
                                label_dir.mkdir(exist_ok=True)
                                local_path = get_image_from_s3(image_obj.s3_key, str(tmpdir))
                                if local_path:
                                    shutil.copy(local_path, label_dir / image_obj.filename)
                                else:
                                    missing_files_count += 1
                            
                            if len(images_to_export) == missing_files_count:
                                st.warning("有効な画像データがなかったため、zipを生成できませんでした。"); return

                            zip_path_str = os.path.join(tmpdir, "export"); shutil.make_archive(zip_path_str, 'zip', dataset_root)
                            zip_filename = f"classification_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                            st.download_button(label="📥 分類データセットをDL", data=Path(f"{zip_path_str}.zip").read_bytes(), file_name=zip_filename, mime="application/zip")
                            if missing_files_count > 0: st.warning(f"{missing_files_count}件のファイルがS3から取得できませんでした。")
                except Exception as e:
                    st.error(f"データセット生成中にエラー: {e}")
                    st.error(traceback.format_exc())
                finally:
                    db.close()

def main():
    st.set_page_config(layout="wide")
    cleanup_old_session_dirs()

    if 'current_page' not in st.session_state:
        st.session_state.current_page = "画像を登録する"
    if 'last_activity_time' not in st.session_state:
        st.session_state.last_activity_time = time.time()
    if 'temp_dir_session' not in st.session_state or not os.path.exists(st.session_state.temp_dir_session):
        st.session_state.temp_dir_session = setup_session_temp_dir()

    if check_session_timeout():
        return

    try:
        logo_path = "./KG-Motors-Logo.png"
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, width=150) 
        else:
            st.sidebar.warning(f"ロゴファイルが見つかりません: {os.path.abspath(logo_path)}")
    except Exception as e_logo:
        st.sidebar.error(f"ロゴ画像の読み込み/表示中にエラー: {e_logo}")

    st.sidebar.title("ナビゲーション")
    
    page_option_register = "画像を登録する"
    page_option_search = "画像を検索する"
    page_option_dataset = "データセット作成"

    if st.sidebar.button(page_option_register, key="nav_button_register", use_container_width=True):
        st.session_state.current_page = page_option_register
        st.session_state.last_activity_time = time.time()
        st.rerun()

    if st.sidebar.button(page_option_search, key="nav_button_search", use_container_width=True):
        st.session_state.current_page = page_option_search
        st.session_state.last_activity_time = time.time()
        st.rerun()

    if st.sidebar.button(page_option_dataset, key="nav_button_dataset", use_container_width=True):
        st.session_state.current_page = page_option_dataset
        st.session_state.last_activity_time = time.time()
        st.rerun()

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = True 

    if st.session_state.current_page == page_option_register:
        st.title("KG画像登録システム") 
        convert_image_to_vector() 
    
    elif st.session_state.current_page == page_option_search:
        run_image_search_app()

    elif st.session_state.current_page == page_option_dataset:
        create_dataset_page()

if __name__ == "__main__":
    main()