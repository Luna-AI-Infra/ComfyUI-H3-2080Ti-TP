# Workflow formats and presets

**Start here for reference images:** [Ref2VA walkthrough / 双参考图接入教程](REF2VA.md), [editor JSON](ui/ref2va-robot-480p-5s.json), [matching API JSON](api/ref2va-robot-480p-5s.json), and [CC0 reference images](../examples/reference/). This example has frontend import/export and native graph validation; no new generation timing is claimed.

- `ui/`: ComfyUI editor-format graphs; drag/open them in the frontend. They were exported from the working deployment, then updated to the current H3 widget order (including VAE fusion and all TP planner fields) and the performance node settings. Their scenes differ from the API examples. Node/schema checks do not replace reviewing the actual frontend import.
- `api/`: node-ID keyed API graphs; send them to `/prompt` using the format below. They are not editor JSON. The5s/10s text-only graphs originate in E60, and the reference graph in E65/E75.

```python
import json
import urllib.request

with open('workflows/api/t2va-480p-5s.json', encoding='utf-8') as f:
    graph = json.load(f)
body = json.dumps({'prompt': graph, 'client_id': 'h3-tp-example'}).encode()
request = urllib.request.Request('http://127.0.0.1:8188/prompt', data=body,
                                 headers={'Content-Type': 'application/json'})
print(urllib.request.urlopen(request).read().decode())
```

Use your server's configured port. For the robot Ref2VA example, copy the bundled `reference-a.png` and `reference-b.png` into the ComfyUI input directory or upload your own images and update the graph. The older `ref2va-480p-5s.json` retains the historical character prompt: provide suitable inputs and adapt its prompt rather than assuming the robot images match that scene. No private reference image is bundled.

The process preset is separate: launch with `configs/performance.json` for the performance environment, or explicitly choose a conservative graph/backend together with `configs/conservative.json`. Changing only the environment does not change node inputs. Pair pipeline/decode precision and cache generation; do not combine unmatched nodes casually.

Multiple HTTP submissions use the normal serial queue. For different samples, change the actual RandomNoise seed, not just a benchmark nonce. `H3LoadBenchmarkNonce` forces selected benchmark work to execute; it does not choose a new generation seed. A queue smoke test is not a multiple-video reliability guarantee.

模型下载、精度开关、显卡编号和后端依赖均需要按自己的环境核对。UI示例与API示例不是同一种JSON格式；私人参考输入未打包，参考图占位名需要替换。
