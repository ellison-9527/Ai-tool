import random

def guess_number_game():
    number_to_guess = random.randint(1, 100)
    print("欢迎来到猜数字游戏！我已经想好了一个1到100之间的数字。")
    
    while True:
        try:
            user_guess = int(input("请输入你的猜测："))
            if user_guess < number_to_guess:
                print("太小了！再试一次。")
            elif user_guess > number_to_guess:
                print("太大了！再试一次。")
            else:
                print(f"恭喜你！正确答案就是 {number_to_guess}。")
                break
        except ValueError:
            print("请输入一个有效的整数。")

if __name__ == "__main__":
    guess_number_game()
