from fastapi import FastAPI
import time
import asyncio
from pydantic import BaseModel, validator
from starlette.requests import Request
from starlette.testclient import TestClient
import uvicorn
import asyncio
import sys

app = FastAPI()

a = []

def _global(num):
    global a
    a.append(num)


print(id(a), a) 
_global(sys.argv[1])
print(id(a), a) 


@app.post('/tt')
def xx(request: Request):
    print(id(a), a)
    return a


@app.post('/tt/{b}')
def yy(request: Request, b: int):
    a.append(b)
    return a[-1]


@app.post('/change')
def zz(request: Request, b:int):
    a.append(b)
    return a[-1]


# db = []

# class City(BaseModel):
#     name: str
#     timezone: str

# @app.get('/')
# def index():
#     return {'key' : 'value'}

# @app.get('/cities')
# def get_cities():
#     return db

# @app.post('/cities')
# def create_city(city: City):
#     db.append(city.dict())
#     return db[-1]

# @app.delete('/cities/{city_id}')
# def delete_city(city_id: int):
#     db.pop(city_id-1)
#     return {}


# # 测试
# client = TestClient(app)

# def test_correct_item():
#     data = {"name": "shampoo", "price": 1.5}
#     resp = client.post("/items/", json=data)
#     assert resp.status_code == 200

# def test_wrong_item():
#     data = {"name": "shampoo", "price": -1.5}
#     resp = client.post("/items/", json=data)
#     assert resp.status_code != 200


if __name__ == "__main__":
    # print(id(a), a) 
    # _global(sys.argv[1])
    # print(id(a), a) 

    uvicorn.run(
        app="fastapi_test2:app",
        host="0.0.0.0",
        port=8090,
        backlog=2048,
        limit_concurrency=200,
        workers=1,
        reload=True,
        # debug=True,
    )
