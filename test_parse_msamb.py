if __name__ == "__main__":
    import requests
    from bs4 import BeautifulSoup

    url = "https://www.msamb.com/ApmcDetail/DataGridBind?commodityCode=07026&apmcCode=null"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.msamb.com/ApmcDetail/APMCPriceInformation"
    }

    response = requests.get(url, headers=headers, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.find_all("tr")

    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) == 7:
            market = cols[0]
            modal_price = cols[6]
            print("Market:", market, "| Modal Price:", modal_price)
