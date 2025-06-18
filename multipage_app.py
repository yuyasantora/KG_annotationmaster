import streamlit as st
import os
from common import cleanup_old_session_dirs, check_session_timeout

# アプリ全体の設定。これは最初に一度だけ、このファイルで呼び出します。
st.set_page_config(
    page_title="KG 画像アノテーションツール",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 初期化・セッション管理 ---
# アプリケーション起動時に一度だけ実行
if 'app_initialized' not in st.session_state:
    cleanup_old_session_dirs()
    st.session_state.app_initialized = True

# 全ページ共通のセッションタイムアウトチェック
if check_session_timeout():
    st.stop()

# サイドバーにロゴを配置。これで全てのページに共通して表示されます。
try:
    logo_path = "./KG-Motors-Logo.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=150)
    else:
        st.sidebar.warning(f"ロゴファイルが見つかりません: {os.path.abspath(logo_path)}")
except Exception as e_logo:
    st.sidebar.error(f"ロゴ画像の読み込み中にエラー: {e_logo}")

# ユーザーを自動的に最初のページにリダイレクトさせる
try:
    st.switch_page("pages/画像を登録する.py")
except:
    # st.switch_pageが使えない、またはページが見つからない場合の表示
    st.title("ようこそ！")
    st.markdown("左のサイドバーから操作したいページを選んでください。")
