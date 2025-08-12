from dotenv import load_dotenv

load_dotenv()

# app.py
# Streamlit + LangChain デモ: 旅行プラン即提案（A/B専門家の切替）
# 実行前に OpenAI API キー を環境変数に設定してください:
#   setx OPENAI_API_KEY "sk-..."  (Windows)
#   export OPENAI_API_KEY="sk-..." (macOS/Linux)

import os
import textwrap
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# ペルソナ（システムメッセージ）定義
# -----------------------------
PERSONAS = {
    "A": {
        "title": "高齢者配慮の国内旅行プランナー",
        "system": (
            "あなたは高齢者に配慮した国内旅行の専門プランナーです。"
            "ユーザーの希望や体力面を踏まえ、"
            "段差回避・休憩頻度・移動の負担を考慮した半日〜1日のプランを、"
            "時間順のタイムライン形式で提案してください。"
            "提案には以下のセクションを含めます：\n"
            "1) 主要動線（移動手段・所要時間）\n"
            "2) 観光・立ち寄り（各場所の見どころと滞在目安）\n"
            "3) 食事候補（近くの候補を2つまで）\n"
            "4) 休憩ポイント（タイミングと目安）\n"
            "5) 雨天代替（1案）\n"
            "各項目は箇条書きで簡潔に。"
        ),
    },
    "B": {
        "title": "費用最適化プランナー（移動効率重視）",
        "system": (
            "あなたは費用対効果と移動効率に強い旅行プランナーです。"
            "ユーザーの条件から、移動距離短縮とコスト抑制を優先しつつ、"
            "満足度の高い半日〜1日のプランを時間順のタイムラインで提案してください。"
            "提案には以下のセクションを含めます：\n"
            "1) 主要動線（最短ルート・所要時間・概算交通費）\n"
            "2) 観光・体験（費用目安と代替案）\n"
            "3) 食事候補（価格帯と並びやすさの目安）\n"
            "4) 節約Tips（最大3点）\n"
            "5) 雨天代替（1案）\n"
            "各項目は箇条書きで簡潔に。"
        ),
    },
}

# -----------------------------
# LangChain LLM 準備
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_llm():
    # 安定出力のため温度は低め。必要に応じて変更可。
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def build_chain(persona_key: str):
    """選択されたペルソナのシステムメッセージでチェーンを生成"""
    persona = PERSONAS.get(persona_key, PERSONAS["A"])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", persona["system"]),
            (
                "user",
                textwrap.dedent(
                    """\
                    以下の条件で旅行プランを作成してください。
                    ユーザー入力:
                    {user_input}

                    出力要件:
                    - 時間順（例: 09:00 出発 → 10:15 到着 → …）
                    - 箇条書きを中心に簡潔に
                    - 固有名は一般的な観光地に留め、過度な断定は避ける
                    - 不明点は仮定を明記
                    """
                ),
            ),
        ]
    )
    llm = get_llm()
    return prompt | llm

# ---------------------------------------------------------
# 重要: 指定の形式に従った関数
# 「入力テキスト」と「ラジオボタンの選択値」を引数にとり、
# LLMの回答テキストを戻り値として返します。
# ---------------------------------------------------------
def generate_plan(user_text: str, persona_key: str) -> str:
    """旅行プラン生成関数（要件準拠）"""
    if not user_text.strip():
        return "入力が空です。内容を入力してください。"
    try:
        chain = build_chain(persona_key)
        res = chain.invoke({"user_input": user_text})
        # res は BaseMessage 互換の可能性あり: str に統一
        return getattr(res, "content", str(res))
    except Exception as e:
        return f"エラーが発生しました: {e}"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="旅行プラン即提案デモ", page_icon="🧭", layout="centered")

st.title("🧭 旅行プラン即提案（LangChain + Streamlit）")
st.caption("ラジオボタンで専門家の振る舞いを選び、テキストを入力して送信すると、LLMがタイムライン形式の旅行プランを提案します。")

with st.expander("ℹ️ アプリ概要・使い方", expanded=True):
    st.markdown(
        """
        **このアプリでできること**
        - 入力フォームに「出発地・滞在日数・興味（例：神社、自然、温泉、グルメ）」などを自由に記入すると、
          LLMが**時間順の旅行プラン**を生成します。
        - ラジオボタンで**専門家の振る舞い**を切り替えできます。  
          - **A**: 高齢者配慮の国内旅行プランナー（段差回避・休憩頻度などに配慮）  
          - **B**: 費用最適化プランナー（移動効率・コスト抑制を重視）
        
        **操作方法**
        1. 画面左のラジオで **A/B** を選択  
        2. 入力フォームに旅行条件を記入  
        3. **「プランを生成」** ボタンをクリック
        
        **例: 入力の雛形**
        ```
        出発地: 東京
        滞在: 半日〜1日
        興味: 神社, 旧市街散策, 甘味
        体力: ゆっくり歩ける程度、階段は少なめ希望
        雨天でも動ける代替案もほしい
        ```
        """
    )

# サイドバー: ペルソナ選択
with st.sidebar:
    st.header("設定")
    persona_key = st.radio(
        "専門家の種類",
        options=["A", "B"],
        index=0,
        format_func=lambda k: f"{k}: {PERSONAS[k]['title']}",
    )
    st.markdown("---")
    st.markdown(
        "**モデル**: gpt-4o-mini（変更したい場合はコード内 `get_llm()` を編集）"
    )

# 入力フォーム
st.subheader("リクエストを入力")
user_text = st.text_area(
    "旅行条件・要望を記入してください（自由記述）",
    height=180,
    placeholder="出発地、日程、興味、体力面、費用感などを書いてください。",
)

# 送信
if st.button("プランを生成", type="primary"):
    with st.spinner("LLMが旅行プランを作成中..."):
        answer = generate_plan(user_text, persona_key)  # ← 要件の関数を利用
    st.divider()
    st.subheader("提案プラン")
    st.markdown(answer)
    st.divider()

# フッター情報
st.caption(
    "※ 提案は一般的な情報に基づく自動生成です。実際の運行状況・営業時間・料金は必ず公式情報でご確認ください。"
)
