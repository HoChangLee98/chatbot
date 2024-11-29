## 자소서 기반 챗봇 만들기

### 가상환경 설정 방법  
requirements.txt를 다운받는다.  
chatbot 프로젝트 폴더를 생성하고 requirements.txt를 넣는다.  
이후 vscode의 터미널 창에서 아래 코드를 하나씩 실행한다.  

```cmd
conda create -n chatbot python=3.11
conda activate chatbot  
pip install -r requirements.txt
```

## Contents
자소서 혹은 포토폴리오를 기반으로 질문에 대해 답변을 하는 챗봇을 생성합니다. 

### 1. pdf 문서 준비
files 폴더에 pdf 형식의 파일들을 준비합니다.  

### 2. local에서 실행
chatbot.py 내에 open_ai_key를 입력합니다.  
가상환경을 활성화시켜준 뒤 터미널 상에서 아래의 코드를 실행합니다.  
```cmd
streamlit run chatbot.py
```
