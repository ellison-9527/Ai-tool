import tkinter as tk
from tkinter import messagebox


class Calculator:
    def __init__(self, master):
        self.master = master
        master.title('计算器')

        # 创建一个输入框来显示计算结果
        self.entry = tk.Entry(master, width=40, borderwidth=5)
        self.entry.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # 定义按钮上的文字
        buttons = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3),
            ('0', 4, 0), ('C', 4, 1), ('=', 4, 2), ('+', 4, 3),
        ]

        # 创建按钮并放置它们
        for (text, row, col) in buttons:
            button = tk.Button(master, text=text, width=10, height= 3,
                               command=lambda t=text: self.click(t))
            button.grid(row=row, column=col)

    def click(self, key):
        if key == '=':
            try:
                result = eval(self.entry.get())
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except Exception as e:
                messagebox.showerror('错误', '无效的输入')
                print(e)
        elif key == 'C':
            self.entry.delete(0, tk.END)
        else:
            self.entry.insert(tk.END, key)


if __name__ == '__main__':
    root = tk.Tk()
    my_calculator = Calculator(root)
    root.mainloop()