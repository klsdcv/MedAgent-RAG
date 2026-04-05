"""DUR 병용금기 API를 LangChain Tool로 래핑.

Interaction Agent가 실시간으로 약물 상호작용을 확인할 때 사용.
"""

import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

API_KEY = os.getenv("DATA_API_KEY")
BASE_URL = "https://apis.data.go.kr/1471000/DURPrdlstInfoService03"


@tool
def check_drug_interaction(drug_name_a: str, drug_name_b: str) -> str:
    """두 약물의 병용금기 여부를 DUR API로 확인합니다.

    Args:
        drug_name_a: 첫 번째 약물명 또는 성분명
        drug_name_b: 두 번째 약물명 또는 성분명

    Returns:
        병용금기 정보 문자열
    """
    url = f"{BASE_URL}/getUsjntTabooInfoList03"
    params = {
        "serviceKey": API_KEY,
        "type": "json",
        "numOfRows": 20,
        "itemName": drug_name_a,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"API 호출 실패: {e}"

    items = data.get("body", {}).get("items", [])
    if not items:
        return f"'{drug_name_a}'에 대한 병용금기 정보를 찾을 수 없습니다."

    # drug_name_b와 관련된 항목 필터링
    matches = []
    drug_b_lower = drug_name_b.lower()
    for item in items:
        mixture_name = (item.get("MIXTURE_ITEM_NAME") or "").lower()
        ingr_name = (item.get("INGR_NAME") or "").lower()
        ingr_name_b = (item.get("MIXTURE_INGR_NAME") or "").lower()

        if drug_b_lower in mixture_name or drug_b_lower in ingr_name or drug_b_lower in ingr_name_b:
            matches.append(item)

    if not matches:
        # 넓은 검색: 전체 결과에서 관련 정보 제공
        results = []
        for item in items[:5]:
            mixture = item.get("MIXTURE_ITEM_NAME", "")
            prohbt = item.get("PROHBT_CONTENT", "")
            if mixture:
                results.append(f"- {drug_name_a} + {mixture}: {prohbt}")

        if results:
            return (
                f"'{drug_name_b}'와의 직접적인 병용금기는 확인되지 않았으나, "
                f"'{drug_name_a}'의 병용금기 목록:\n" + "\n".join(results)
            )
        return f"'{drug_name_a}'와 '{drug_name_b}'의 병용금기 정보를 찾을 수 없습니다."

    results = []
    for item in matches:
        mixture = item.get("MIXTURE_ITEM_NAME", "")
        prohbt = item.get("PROHBT_CONTENT", "")
        results.append(f"- {drug_name_a} + {mixture}: {prohbt}")

    return f"병용금기 확인 결과:\n" + "\n".join(results)


@tool
def search_dur_by_ingredient(ingredient_name: str) -> str:
    """성분명으로 DUR 정보를 조회합니다.

    Args:
        ingredient_name: 의약품 성분명

    Returns:
        해당 성분의 DUR 정보
    """
    url = f"{BASE_URL}/getUsjntTabooInfoList03"
    params = {
        "serviceKey": API_KEY,
        "type": "json",
        "numOfRows": 10,
        "ingrName": ingredient_name,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"API 호출 실패: {e}"

    items = data.get("body", {}).get("items", [])
    if not items:
        return f"'{ingredient_name}'에 대한 DUR 정보를 찾을 수 없습니다."

    results = []
    for item in items[:10]:
        item_name = item.get("ITEM_NAME", "")
        mixture = item.get("MIXTURE_ITEM_NAME", "")
        prohbt = item.get("PROHBT_CONTENT", "")
        results.append(f"- {item_name} + {mixture}: {prohbt}")

    return f"'{ingredient_name}' DUR 정보:\n" + "\n".join(results)
