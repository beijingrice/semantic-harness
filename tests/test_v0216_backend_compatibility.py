"""API contract/negotiation tests: offline, no LM Studio and no token usage."""
import json
from dataclasses import replace

import httpx
import pytest

from app.backends import BackendError, _schema_rejected
from app.config import Settings
from app.llm import ChatClient, LLMError
from app.models import ParagraphExtractedDraft, ParagraphReviewDraft


def mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def reply(text='{"units":[]}'):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})


def test_legacy_schema_backward_compatibility():
    requests = []
    def handler(r):
        requests.append(json.loads(r.content))
        return reply()
    transport = mock(handler)
    for active in (False, True):
        ChatClient(Settings(json_schema=active), transport).structured(
            'primary', 'sys', 'user', ParagraphExtractedDraft, 'paragraph_extraction')
    assert 'response_format' not in requests[0]
    assert requests[1]['response_format']['type'] == 'json_schema'
    transport.close()


def test_auto_schema_downgrades_on_explicit_unsupported_error_once_and_caches():
    seen=[]
    def handler(r):
        data=json.loads(r.content)
        seen.append(data)
        if 'response_format' in data:
            return httpx.Response(400, json={'error': {'message': 'response_format json_schema is not supported'}})
        return reply()
    transport=mock(handler)
    cli=ChatClient(Settings(schema_mode='auto', model='p', generation_recovery=False), transport)
    for _ in range(2):
        assert not cli.structured('p','S','U',ParagraphExtractedDraft,'paragraph_extraction').units
    assert [bool('response_format' in x) for x in seen] == [True,False,False]
    assert cli.trace_snapshot()[-1]['schema_mode'] == 'prompt_only'
    transport.close()


def test_schema_on_does_not_silently_downgrade():
    seen=[]
    def handler(r):
        seen.append(1)
        return httpx.Response(400,json={'error':{'message':'response_format not supported'}})
    transport=mock(handler)
    with pytest.raises(LLMError,match='LM_SCHEMA_MODE=auto or off'):
        ChatClient(Settings(schema_mode='on'),transport).structured('p','','',ParagraphExtractedDraft,'paragraph_extraction')
    assert len(seen)==1
    transport.close()


@pytest.mark.parametrize('status,message',[(401,'response_format not supported'),(429,'response_format not supported'),
                                           (500,'response_format unsupported'),(400,'bad model id'),(422,'invalid prompt')])
def test_auto_never_downgrades_on_auth_rate_limit_server_or_ambiguous_bad_request(status,message):
    calls=[]
    def handler(r):
        calls.append(1)
        return httpx.Response(status,json={'error':{'message':message}})
    transport=mock(handler)
    with pytest.raises(LLMError,match='模型 API 请求失败'):
        ChatClient(Settings(schema_mode='auto'),transport).structured('p','','',ParagraphExtractedDraft,'paragraph_extraction')
    assert len(calls)==1
    transport.close()


def test_prompt_only_still_enforces_pydantic_and_fail_closed():
    count=[]
    def handler(r):
        assert 'response_format' not in json.loads(r.content)
        count.append(1)
        return reply('{"units":[{"kind":"concept","claim":5,"paragraph_ids":["en_p1"]}]}')
    transport=mock(handler)
    with pytest.raises(LLMError,match='并非 FAIL'):
        ChatClient(Settings(schema_mode='off'),transport).structured('p','','',ParagraphExtractedDraft,'paragraph_extraction')
    assert len(count)==3
    transport.close()


def test_separate_primary_verifier_endpoints_and_authorization():
    seen=[]
    def handler(r):
        seen.append((str(r.url),r.headers.get('authorization',''),json.loads(r.content)['model']))
        return reply('{"units":[]}' if str(r.url).startswith('https://primary') else
                     '{"verdict":"PASS","reason":"same","issues":[]}')
    transport=mock(handler)
    cfg=Settings(base_url='https://primary/v1',api_key='key-PRIMARY',model='qwen',
                 verifier_base_url='https://review/v1',verifier_api_key='key-REVIEW',
                 verifier_model='gemma',schema_mode='off')
    cli=ChatClient(cfg,transport,transport)
    cli.choose_model();cli.choose_verifier()
    cli.structured('qwen','','',ParagraphExtractedDraft,'paragraph_extraction')
    cli.structured('gemma','','',ParagraphReviewDraft,'paragraph_review')
    assert seen[0]==('https://primary/v1/chat/completions','Bearer key-PRIMARY','qwen')
    assert seen[1]==('https://review/v1/chat/completions','Bearer key-REVIEW','gemma')
    transport.close()


def test_model_discovery_policies():
    seen=[]
    def handler(r):
        seen.append(str(r.url))
        return httpx.Response(404)
    transport=mock(handler)
    for mode in ('optional','off'):
        cli=ChatClient(Settings(model_discovery=mode),transport)
        assert cli.validate_model_access('anything','other')
    assert len(seen)==1  # optional tries once, off never
    with pytest.raises(LLMError,match='discovery failed'):
        ChatClient(Settings(model_discovery='required'),transport).validate_model_access('anything')
    transport.close()


def test_optional_model_discovery_does_not_ignore_known_absent_id():
    def handler(r):
        return httpx.Response(200,json={'data':[{'id':'other'}]})
    transport=mock(handler)
    with pytest.raises(LLMError,match='not in provider'):
        ChatClient(Settings(model_discovery='optional'),transport).validate_model_access('missing')
    transport.close()


def test_custom_non_openai_adapter_is_independent_of_wire_format():
    def handler(r):
        assert str(r.url).endswith('/generate')
        body=json.loads(r.content)
        assert 'response_format' not in body and body['system']=='instructions'
        assert r.headers['x-api-key']=='secret'
        return httpx.Response(200,json={'output':{'text':'{"units":[]}'},'usage':{'input_tokens':7,'output_tokens':3}})
    transport=mock(handler)
    cli=ChatClient(Settings(backend='examples.custom_backend_example:ExampleNonOpenAIBackend',
                            base_url='https://custom',api_key='secret',model='p',
                            schema_mode='off',model_discovery='off'),transport)
    assert not cli.structured('p','instructions','hello',ParagraphExtractedDraft,'paragraph_extraction').units
    assert cli.trace_snapshot()[0]['prompt_tokens']==7
    assert cli.validate_model_access('p')=='skipped'
    transport.close()


def test_unrecognized_adapter_or_bad_options_fail_on_construction():
    with pytest.raises(BackendError):
        ChatClient(Settings(backend='no_such_backend'))
    with pytest.raises(ValueError):
        ChatClient(Settings(schema_mode='broken'))
    with pytest.raises(ValueError):
        ChatClient(Settings(model_discovery='broken'))


def test_backend_error_redacts_authorization_response_body_and_prompts():
    def handler(r):
        return httpx.Response(401,text='Authorization: Bearer secret-user-token; original sensitive text')
    transport=mock(handler)
    with pytest.raises(LLMError) as exc:
        ChatClient(Settings(api_key='secret-user-token'),transport).structured(
            'p','private instructions','sensitive text',ParagraphExtractedDraft,'paragraph_extraction')
    assert 'secret-user-token' not in str(exc.value)
    assert 'sensitive text' not in str(exc.value)
    transport.close()


def test_schema_rejection_classifier_does_not_accept_every_http400():
    assert _schema_rejected(400,'Invalid response_format: json_schema unsupported')
    assert not _schema_rejected(400,'Bad request body')
    assert not _schema_rejected(401,'response_format unsupported')


def test_auto_schema_fallback_cache_is_per_model_even_same_backend():
    sent=[]
    def handler(r):
        d=json.loads(r.content)
        sent.append((d['model'], 'response_format' in d))
        if d['model']=='no_schema' and 'response_format' in d:
            return httpx.Response(422,json={'error':{'message':'json_schema not supported'}})
        return reply()
    http=mock(handler)
    client=ChatClient(Settings(schema_mode='auto'),http)
    client.structured('no_schema','','',ParagraphExtractedDraft,'paragraph_extraction')
    client.structured('has_schema','','',ParagraphExtractedDraft,'paragraph_extraction')
    client.structured('no_schema','','',ParagraphExtractedDraft,'paragraph_extraction')
    assert sent==[('no_schema',True),('no_schema',False),('has_schema',True),('no_schema',False)]
    http.close()


def test_bad_native_schema_request_not_automatically_silenced():
    transport=mock(lambda r:httpx.Response(400,json={'error':{'message':'invalid response_format json_schema definition'}}))
    with pytest.raises(LLMError,match='模型 API 请求失败'):
        ChatClient(Settings(schema_mode='auto'),transport).structured('p','','',ParagraphExtractedDraft,'paragraph_extraction')
    transport.close()
