import torch


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('best.pth', map_location=device)
    model.eval()

    with torch.no_grad():
        while True:
            a = input()
            a, b = a.split(" ")
            x = torch.tensor([[[float(a)], [float(b)]]], dtype=torch.float32)
            x = x.to(device)
            y = model(x)
            print("predict:", y.item())
            print("real:", float(a) + float(b))
