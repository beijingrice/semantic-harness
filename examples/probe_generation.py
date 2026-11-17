"""Opt-in four-request diagnostic only. Not an accuracy benchmark or an automatic test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from app.config import Settings
from app.llm import ChatClient, LLMError, parse_json_object
from app.models import ParagraphExtractedDraft
from app.paragraphs import build_paragraphs, compact_paragraphs, verify_ids
from app.prompts import EXTRACTION_SYSTEM_PARAGRAPH, paragraph_extract_user


def main():
    parser=argparse.ArgumentParser(description='4 次短文本 LM Studio Schema ON/OFF 比较；不运行评测集')
    parser.add_argument('--out',type=Path,default=Path('probe_v028_generation.json'))
    args=parser.parse_args()
    settings=Settings()
    client=ChatClient(settings)
    examples=[('English', 'If more than 12% of mandatory fields are blank, reject the response.', 'en'),
              ('Chinese', '若必填字段缺失比例超过12%，应拒绝该答卷。', 'zh')]
    trials=[]
    try:
        model=client.choose_model()
        if model not in client.models():
            raise LLMError(f'未找到主模型: {model}')
        for language, text, prefix in examples:
            paragraphs=build_paragraphs(text,prefix)
            user=paragraph_extract_user(text,language,compact_paragraphs(paragraphs),settings.max_keywords)
            for mode in ('json_schema','prompt_only'):
                payload={'model':model,'messages':[{'role':'system','content':EXTRACTION_SYSTEM_PARAGRAPH},
                    {'role':'user','content':user}],'temperature':0,
                    'max_tokens':settings.max_tokens,'stream':False}
                if mode=='json_schema':
                    payload['response_format']={'type':'json_schema','json_schema':{
                        'name':'paragraph_extraction','strict':True,
                        'schema':ParagraphExtractedDraft.model_json_schema()}}
                client.start_trace()
                error=None
                try:
                    raw=client._completion(payload=payload,model=model,
                        schema_name='paragraph_extraction',attempt=1,recovery=False)
                    draft=ParagraphExtractedDraft.model_validate(parse_json_object(raw))
                    for unit in draft.units:
                        verify_ids(unit.paragraph_ids,paragraphs)
                    client.request_trace[-1]['outcome']='validated'
                except (LLMError,ValueError) as exc:
                    error=str(exc)[:350]
                trial={'language':language,'mode':mode,'valid':error is None,
                       'error':error,'requests':client.trace_snapshot()}
                trials.append(trial)
                metric=trial['requests'][0]
                print(f'{language:<8} {mode:<12} valid={trial["valid"]} '
                      f'finish={metric.get("finish_reason")} '
                      f'reasoning_tokens={metric.get("reasoning_tokens")} '
                      f'completion_tokens={metric.get("completion_tokens")} '
                      f'duration_ms={metric.get("duration_ms")}',flush=True)
    finally:
        client.close()
    args.out.write_text(json.dumps({'model':model,'trials':trials},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('报告：',args.out.resolve())

if __name__=='__main__':
    main()
