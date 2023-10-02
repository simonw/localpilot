import config
import httpx
import json
import os
import subprocess
import logging
from starlette import applications, responses, exceptions
from starlette.requests import Request

app = applications.Starlette()
state = config.models[config.models['default']]
local_server_process = None
model_directory = os.path.expanduser("~/models")
logging.basicConfig(level=logging.DEBUG)


def start_local_server(model_filename):
    global local_server_process
    if local_server_process:
        local_server_process.terminate()
        local_server_process.wait()
    cmd = ["python3", "-m", "llama_cpp.server", "--model", model_filename,
           "--n_gpu_layers", "1", "--n_ctx", "4096"]  # TODO: set this more correctly
    logging.debug('Running: %s' % ' '.join(cmd))
    local_server_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@app.route('/set_target', methods=['POST'])
async def set_target(request: Request):
    global state
    response = await request.json()
    target = response['target']
    if target not in config.models:
        raise exceptions.HTTPException(
            status_code=400, detail=f'Invalid target: {target}')

    state = config.models[target]
    if config.models[target].get("type") == "local":
        start_local_server(os.path.join(
            model_directory, config.models[target]['filename']))

    message = f'Target set to {state}'
    return responses.JSONResponse({'message': message}, status_code=200)


@app.route('/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def proxy(request: Request):
    global state
    path = request.url.path
    logging.debug(f'Current state: {state}')

    if state['type'] == 'remote':
        url = f"{state['domain']}{path}"
    elif state['type'] == 'local':
        url = f"http://localhost:8000{path}"

    data = await request.body()

    try:
        decoded = json.loads(data)
        if isinstance(decoded, dict) and "prompt" in decoded:
            prompt = decoded.pop("prompt", "")
            suffix = decoded.pop("suffix", "")
            print("------------------")
            print(json.dumps(decoded, indent=4))
            print("------\nPrompt:\n\n")
            print(prompt)
            print("------\nSuffix:\n\n")
            print(suffix)
            print("------------------")
            print("")

    except json.JSONDecodeError:
        pass

    headers = dict(request.headers)
    r = None
    async with httpx.AsyncClient() as client:
        try:
            if request.method == 'GET':
                r = await client.get(url, params=request.query_params, headers=headers)
            elif request.method == 'POST':
                r = await client.post(url, data=data, headers=headers, timeout=30)
            elif request.method == 'PUT':
                r = await client.put(url, data=data, headers=headers)
            elif request.method == 'DELETE':
                r = await client.delete(url, headers=headers)
        except httpx.RemoteProtocolError as exc:
            logging.debug(f'Connection closed prematurely: {exc}')
    content = r.content if r else ''
    status_code = r.status_code if r else 204
    headers = dict(r.headers) if r else dict()
    print("----")
    print("Response headers:")
    print(json.dumps(headers, indent=4))
    print("Response body:")
    # If content is data: lines, reformat it
    debug_content = content
    try:
        lines = content.decode().split("\n")
        if any(line.startswith("data: ") for line in lines):
            bits = []
            for line in lines:
                if line.startswith("data: "):
                    try:
                        bits.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        bits.append(line)
                else:
                    bits.append(line)
            debug_content = json.dumps(bits, indent=4)
    except Exception:
        pass
    print(debug_content)
    print("------------------")
    return responses.Response(content=content, status_code=status_code, headers=headers)


@app.exception_handler(404)
async def not_found(request, exc):
    return responses.JSONResponse({"error": "Not found"}, status_code=404)


@app.exception_handler(500)
async def server_error(request, exc):
    return responses.JSONResponse({"error": "Server error"}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
