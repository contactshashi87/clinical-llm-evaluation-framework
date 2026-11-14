import asyncio

import pandas as pd
from .settings import ROOT, OUTPUT_DIR
from .chatbot_executor import ChatbotExecutor


async def main():
    data=pd.read_csv(ROOT/'data/chatbot_app/multiturn_regression.csv')
    results=[]
    ex = ChatbotExecutor()
    try:
        await ex._ensure_page()
        for _,row in data.iterrows():
            answers=[]
            ok=True; err=''
            # Playwright keeps one page/session, which preserves conversation context.
            for turn in [x.strip() for x in row.conversation.split('|')]:
                q=turn.split(':',1)[1].strip() if ':' in turn else turn
                r=await ex.ask(q); answers.append(r.answer)
                if r.status!='PASS': ok=False; err=r.error
            results.append({**row.to_dict(),'actual_conversation':'\n'.join(answers),'status':'EXECUTED' if ok else 'ERROR','error':err})
    finally:
        await ex.close()
    out=pd.DataFrame(results); out.to_csv(OUTPUT_DIR/'multiturn_regression_results.csv',index=False)
    print(out[['id','status']].to_string(index=False))
    return out


if __name__=='__main__':
    asyncio.run(main())
