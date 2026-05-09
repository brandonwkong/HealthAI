from langgraph_agent import app, State

def main():
    print("Health agent CLI. Type 'quit' to exit.\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"quit", "exit"}:
            break

        state: State = {"user_message": msg}

        # keep looping until we get final output
        while True:
            out = app.invoke(state)

            if out.get("final"):
                print(f"\nAssistant:\n{out['final']}\n")
                break

            if out.get("follow_up_question"):
                q = out["follow_up_question"]
                print(f"\nAssistant (question): {q}\n")
                ans = input("You (answer): ").strip()

                # IMPORTANT: carry forward the whole state (including intake)
                state = {
                    "user_message": msg,
                    "intake": out.get("intake", {}),
                    "context": out.get("context", ""),
                    "follow_up_question": q,
                    "follow_up_answer": ans,
                }
                continue

            # safety fallback (shouldn't happen)
            print("\nAssistant:\nI need a bit more information to continue.\n")
            break

if __name__ == "__main__":
    main()
