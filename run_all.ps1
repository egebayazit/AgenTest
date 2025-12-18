Write-Host "Launching AgenTest services..." -ForegroundColor Green

# --- 0) Ollama ---
Start-Process powershell -ArgumentList " -NoExit -Command `
ollama serve
"

# --- 1) llm_service ---
Start-Process powershell -ArgumentList " -NoExit -Command `
cd '$env:USERPROFILE\Desktop\AgenTest\agent-llm'; `
conda activate base; `
uvicorn llm_service:app --host 0.0.0.0 --port 18888 --reload
"

# --- 2) controller_service ---
Start-Process powershell -ArgumentList " -NoExit -Command `
cd '$env:USERPROFILE\Desktop\AgenTest\agent-controller'; `
conda activate base; `
`$env:SUT_STATE_URL='http://192.168.137.249:18080/state'; `
uvicorn controller_service:app --host 0.0.0.0 --port 18800 --reload
"

# --- 3) OmniParser ---
Start-Process powershell -ArgumentList " -NoExit -Command `
cd '$env:USERPROFILE\Desktop\AgenTest\agent-ods\OmniParser'; `
conda activate omni; `
`$env:OMNI_NO_FLASH_STUB='1'; `
`$env:TRANSFORMERS_NO_FLASH_ATTENTION='1'; `
`$env:TRANSFORMERS_ATTENTION_IMPLEMENTATION='sdpa'; `
python -m omnitool.omniparserserver.omniparserserver --som_model_path .\weights\icon_detect\model.pt --caption_model_name florence2 --caption_model_path .\weights\icon_caption_florence --device cuda
"

# --- 4) UI (Vite Dev Server) ---
Start-Process powershell -ArgumentList " -NoExit -Command `
cd '$env:USERPROFILE\Desktop\AgenTest\agent-llm\ui'; `
npm run dev
"

Write-Host "All services launched!" -ForegroundColor Yellow