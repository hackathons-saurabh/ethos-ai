#!/usr/bin/env python3
"""
Test script to verify frontend-backend integration
"""
import requests
import json
import time

def test_backend_health():
    """Test if backend is running and healthy"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend at http://localhost:8080")
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint that frontend uses"""
    try:
        payload = {
            "message": "Hello, this is a test message",
            "scenario": "hiring",
            "ethos_enabled": True
        }
        
        response = requests.post(
            "http://localhost:8080/chat",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Chat endpoint working!")
            result = response.json()
            print(f"Response: {result}")
            return True
        else:
            print(f"❌ Chat endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing chat endpoint: {e}")
        return False

def test_file_upload():
    """Test file upload endpoint"""
    try:
        # Create a simple test CSV
        test_csv = "name,age,gender,target\nJohn,25,male,1\nJane,30,female,0"
        
        files = {'file': ('test.csv', test_csv, 'text/csv')}
        data = {'name': 'test_dataset', 'target_column': 'target'}
        
        response = requests.post(
            "http://localhost:8080/upload/dataset",
            files=files,
            data=data,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ File upload endpoint working!")
            result = response.json()
            print(f"Upload result: {result}")
            return True
        else:
            print(f"❌ File upload returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing file upload: {e}")
        return False

def test_frontend_access():
    """Test if frontend is accessible"""
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible!")
            return True
        else:
            print(f"❌ Frontend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to frontend at http://localhost:3000")
        return False
    except Exception as e:
        print(f"❌ Error testing frontend: {e}")
        return False

def main():
    print("🔍 Testing Frontend-Backend Integration")
    print("=" * 50)
    
    # Test backend health
    print("\n1. Testing Backend Health...")
    backend_ok = test_backend_health()
    
    # Test chat endpoint
    print("\n2. Testing Chat Endpoint...")
    chat_ok = test_chat_endpoint()
    
    # Test file upload
    print("\n3. Testing File Upload...")
    upload_ok = test_file_upload()
    
    # Test frontend access
    print("\n4. Testing Frontend Access...")
    frontend_ok = test_frontend_access()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 INTEGRATION SUMMARY")
    print("=" * 50)
    
    if backend_ok and chat_ok and upload_ok and frontend_ok:
        print("🎉 ALL TESTS PASSED! Frontend and Backend are properly integrated.")
        print("\n✅ You can now:")
        print("   - Start your frontend and it will connect to backend")
        print("   - Use the chat interface in the frontend")
        print("   - Upload datasets through the frontend")
        print("   - Run demos that will process through the backend")
    else:
        print("❌ Some tests failed. Check the issues above.")
        
        if not backend_ok:
            print("\n🔧 To fix backend issues:")
            print("   - Make sure backend container is running: docker-compose up backend")
            print("   - Check backend logs: docker logs ethos-backend")
            
        if not frontend_ok:
            print("\n🔧 To fix frontend issues:")
            print("   - Make sure frontend container is running: docker-compose up frontend")
            print("   - Check frontend logs: docker logs ethos-frontend")

if __name__ == "__main__":
    main() 