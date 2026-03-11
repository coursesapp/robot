import traceback

from main import execute_agent_tool


def run_test_case(name, payload, expect_error=False):
    print(f"\n=== Test: {name} ===")
    print(f"Payload: {payload}")
    try:
        result = execute_agent_tool(payload)
        print(f"Tool returned: {result!r}")
        if expect_error:
            print(f"[FAIL] Expected an error, but call succeeded.")
            return False
        print("[PASS]")
        return True
    except Exception as e:
        print(f"Exception raised: {e!r}")
        traceback.print_exc()
        if expect_error:
            print("[PASS] (error was expected)")
            return True
        print("[FAIL] (unexpected error)")
        return False


def main():
    test_cases = []

    # --- 1) MathModule-style tests (using execute_python_code for math) ---
    math_snippets = [
        "print(2 + 2)",
        "print(10 * 5)",
        "print((3 ** 3) - 5)",
        "import math\nprint(math.sqrt(144))",
        "nums = [1, 2, 3, 4]\nprint(sum(nums) / len(nums))",
    ]
    for i, snippet in enumerate(math_snippets, start=1):
        test_cases.append(
            (
                f"MathModule #{i} - math via execute_python_code",
                {
                    "tool_choice": "execute_python_code",
                    "arguments": {"code_string": snippet},
                },
                False,
            )
        )

    # --- 2) PythonScript-style tests (general scripting via execute_python_code) ---
    script_snippets = [
        "for i in range(3):\n    print('loop', i)",
        "def greet(name):\n    print('hello', name)\n\ngreet('robot')",
        "data = {'a': 1, 'b': 2}\nprint(list(data.keys()))",
        "try:\n    1/0\nexcept ZeroDivisionError:\n    print('caught error')",
        "values = [x*x for x in range(5)]\nprint(values)",
    ]
    for i, snippet in enumerate(script_snippets, start=1):
        test_cases.append(
            (
                f"PythonScript #{i} - execute_python_code",
                {
                    "tool_choice": "execute_python_code",
                    "arguments": {"code_string": snippet},
                },
                False,
            )
        )

    # --- 3) Weather tests ---
    weather_locations = [
        "Cairo, Egypt",
        "New York, USA",
        "Tokyo, Japan",
        "Berlin, Germany",
    ]
    for i, loc in enumerate(weather_locations, start=1):
        test_cases.append(
            (
                f"Weather #{i} - location '{loc}'",
                {
                    "tool_choice": "get_current_weather",
                    "arguments": {"location": loc},
                },
                False,
            )
        )
    # 5th weather test: missing location (should error)
    test_cases.append(
        (
            "Weather #5 - missing location",
            {
                "tool_choice": "get_current_weather",
                "arguments": {},
            },
            True,
        )
    )

    # --- 4) Deep search tests ---
    deep_search_queries = [
        "latest news about artificial intelligence",
        "humanoid robots safety best practices",
        "weather in Cairo today",
        "OpenAI GPT research 2026",
    ]
    for i, q in enumerate(deep_search_queries, start=1):
        test_cases.append(
            (
                f"DeepSearch #{i} - query '{q}'",
                {
                    "tool_choice": "perform_deep_search",
                    "arguments": {"query": q},
                },
                False,
            )
        )
    # 5th deep-search test: missing query (should error)
    test_cases.append(
        (
            "DeepSearch #5 - missing query",
            {
                "tool_choice": "perform_deep_search",
                "arguments": {},
            },
            True,
        )
    )

    # --- 5) Hardware tests ---
    hardware_cases = [
        ("Hardware #1 - right_arm raise", "right_arm", "raise"),
        ("Hardware #2 - left_arm lower", "left_arm", "lower"),
        ("Hardware #3 - head nod", "head", "nod"),
        ("Hardware #4 - torso turn_left", "torso", "turn_left"),
        ("Hardware #5 - torso turn_right", "torso", "turn_right"),
    ]
    for name, body_part, action in hardware_cases:
        test_cases.append(
            (
                name,
                {
                    "tool_choice": "control_hardware_motors",
                    "arguments": {"body_part": body_part, "action": action},
                },
                False,
            )
        )

    # --- 6) Router / payload edge cases (extra safety checks) ---
    test_cases.extend(
        [
            (
                "Router - unknown tool_choice",
                {"tool_choice": "non_existent_tool", "arguments": {}},
                False,  # should not error; should return error string
            ),
            (
                "Router - missing arguments key",
                {"tool_choice": "get_current_weather"},
                True,  # KeyError expected
            ),
            (
                "Router - missing tool_choice key",
                {"arguments": {"location": "Cairo, Egypt"}},
                True,
            ),
        ]
    )

    total = len(test_cases)
    passed = 0

    print(f"Running {total} test cases against execute_agent_tool...\n")

    for name, payload, expect_error in test_cases:
        if run_test_case(name, payload, expect_error=expect_error):
            passed += 1

    print("\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}")


if __name__ == "__main__":
    main()

