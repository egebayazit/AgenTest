# run service  
uvicorn llm_service:app --host 0.0.0.0 --port 18888 --reload  
# run ui  
npm run dev
# ollama
Make sure Ollama is running: ollama serve 
$env:OLLAMA_DEBUG="1"
ollama serve
-------------------------------------
# start llama server not used with ollama
 cd C:\Users\hazal\Desktop\AgenTest\agent-llm\llama-server
 .\start_server.bat(llama server)

 nvidia-smi -l 1 --query-gpu=timestamp,name,pstate,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv > C:\Users\hazal\Desktop\gpu_detailed_log.csv
