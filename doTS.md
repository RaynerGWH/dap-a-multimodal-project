ssh first, then:

pkill -f jupyter && uvicorn server:app --host 0.0.0.0 --port 8888

