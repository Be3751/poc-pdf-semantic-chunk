import requests
import base64

def main():
    pdf = load_pdf('example.pdf')
    try:
        tokenized = tokenize(pdf)
        print(f'Tokenized data: {tokenized}')
    except ValueError as e:
        print(f'Error: {e}')

def load_pdf(file: str) -> str:
    with open(file, 'rb') as f:
        pdf_content = f.read()
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    return pdf_base64

def tokenize(data: str) -> dict:
    payload = {
        'content': {
            '$content-type': 'application/pdf',
            '$content': data
        },
        'documentType': 'PDF',
        'fileName': 'example.pdf',
        'splittingStrategy': 'RECURSIVE',
        'secondarySplittingStrategy': 'RECURSIVE',
        'chunkSize': 512,
        'chunkOverlap': 128
    }

    url = 'http://localhost:7071/api/tokenize_trigger'
    headers = {'Content-Type': 'application/json'}

    response = requests.get(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise ValueError(f'Error: {response.status_code}')
    return response.json()

if __name__ == '__main__':
    main()
