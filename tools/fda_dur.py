"""FDA DUR (Drug Utilization Review) API 호출 툴.

원본의 `src/tools/dur_api.py`를 그대로 포팅. 약물 상호작용·금기·임부 사용
정보 조회용.
"""

import httpx

from config.settings import settings


def lookup_interaction(drug_a: str, drug_b: str, timeout: float = 10.0) -> dict:
    """두 약물의 상호작용 정보를 FDA OpenAPI에서 조회.

    Returns:
        FDA 응답 dict. 결과가 없으면 {"results": []}.
    """
    url = f"{settings.FDA_DUR_BASE_URL}/event.json"
    params = {
        "search": f'patient.drug.openfda.generic_name:"{drug_a}"+AND+patient.drug.openfda.generic_name:"{drug_b}"',
        "limit": 5,
    }
    try:
        resp = httpx.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return {"results": []}
