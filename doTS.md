ssh first, then:

pkill -f jupyter && uvicorn server:app --host 0.0.0.0 --port 8000

# kill processes on port 8888
pkill -f "jupyter.*8888"