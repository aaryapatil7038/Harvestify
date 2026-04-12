if __name__ == "__main__":
    import requests

    url = "https://www.msamb.com/ApmcDetail/DataGridBind?commodityCode=07026&apmcCode=null"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.msamb.com/ApmcDetail/APMCPriceInformation"
    }

    response = requests.get(url, headers=headers, timeout=30)

    print("Status code:", response.status_code)
    print("Response text:")
    print(response.text[:2000])
