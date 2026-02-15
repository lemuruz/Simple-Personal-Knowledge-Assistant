import streamlit
import requests

def is_cloud_auth_available():
    return hasattr(streamlit, "user") and hasattr(streamlit.user, "is_logged_in")

def login():
    if is_cloud_auth_available():
        if not streamlit.user.is_logged_in:
            if streamlit.button("Log in"):
                streamlit.login()
        else:
            if streamlit.button("Log out"):
                streamlit.logout()
            streamlit.write(f"Hello, {streamlit.user.name}!")
    else:
        if "logged_in" not in streamlit.session_state:
            streamlit.session_state.logged_in = False
        
        if not streamlit.session_state.logged_in:
            if streamlit.button("Log in"):
                streamlit.session_state.logged_in = True
                streamlit.rerun()
            streamlit.stop()
        else:
            if streamlit.button("Log out"):
                streamlit.session_state.logged_in = False
                streamlit.rerun()
            streamlit.write("Hello, User!")



def send_prompt(prompt):
    username = streamlit.user.name if is_cloud_auth_available() and streamlit.user.is_logged_in else "TemporaryUser"
    url = "http://localhost:8000/prompt"
    data = {
        "Username": username,
        "prompt": prompt
    }
    response = requests.post(url, json=data)
    return {
        "username": username,
        "response": response.json()
    }

def main():
    streamlit.title("=== Your Personal AI Chef ===")
    prompt = streamlit.text_input("Enter a prompt")
    if streamlit.button("Send Prompt"):
        if prompt:
            with streamlit.spinner("Generating response..."):
                response = send_prompt(prompt)
            streamlit.write("AI Chef's Response:")
            streamlit.write(response)
        else:
            streamlit.warning("Please enter a prompt before sending.")

if __name__ == "__main__":    
    main()
