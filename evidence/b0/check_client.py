"""Exercise benchmark client on token-ID SSE responses before using NPU results."""
import asyncio
import json

import aiohttp
from aiohttp import web

import run_b0 as b


async def main():
    async def endpoint(req):
        p=await req.json()
        assert p['prompt']==[123]*40000
        assert p['add_special_tokens'] is False
        response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
        await response.prepare(req)
        frames=[{'choices':[{'text':'A','token_ids':[32], 'prompt_token_ids':p['prompt']}]},
                {'choices':[{'text':' B','token_ids':[33,34],'finish_reason':'length'}]},
                {'choices':[],'usage':{'prompt_tokens':40000,'completion_tokens':3}}]
        for frame in frames:
            await response.write(('data: '+json.dumps(frame)+'\n\n').encode())
        await response.write(b'data: [DONE]\n\n')
        await response.write_eof()
        return response
    app=web.Application(client_max_size=10<<20)
    app.router.add_post('/v1/completions',endpoint)
    runner=web.AppRunner(app);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    port=site._server.sockets[0].getsockname()[1]
    b.URL=f'http://127.0.0.1:{port}'
    try:
        async with aiohttp.ClientSession(read_bufsize=4<<20) as session:
            row={'id':'test','input_tokens':40000,'output_tokens':3,'prompt_token_ids':[123]*40000}
            result=await b.request(session,row,False)
            assert result['success'],result
            assert result['output_token_ids']==[32,33,34] and result['text']=='A B'
            row['output_tokens']=4
            failed=await b.request(session,row,False)
            assert not failed['success'] and 'short output' in failed['error']
            row['input_tokens']=39999
            failed=await b.request(session,row,False)
            assert not failed['success'] and 'prompt token mismatch' in failed['error']
        print('PASS: large SSE lines, multi-token chunks, usage checks, short-output and prompt-mismatch rejection')
    finally:await runner.cleanup()


if __name__=='__main__':asyncio.run(main())
