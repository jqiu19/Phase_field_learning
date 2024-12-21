# Open a file in write mode
with open("love_message.txt", "w", encoding="utf-8") as file:
    # Write the phrase 1000 times
    for _ in range(1000):
        file.write("张雪芳我爱你！！！\n")

print("File written successfully with 1000 lines!")
