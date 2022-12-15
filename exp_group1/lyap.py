import torch
    

@torch.no_grad()
def lyapunov_exponent(model, device, max_iter = 1000):
    model.to(device)
    LEs = torch.zeros(model.n).to(device)
    h = torch.ones(model.n).to(device)
    Q = torch.eye(model.n).to(device)

    for _ in range(max_iter):
        h = model.step(h)
        Df = model.d_step(h)
        A = Df @ Q
        Q, R = torch.linalg.qr(A)
        LEs += torch.log(torch.abs(torch.diag(R)))
    
    LEs = LEs / max_iter
    return LEs

if __name__ == "__main__":
    import models
    import timeit

    model = models.RNN(2, 200, 2, 5)
    model.w.data = torch.eye(200) * 2
    model.b.data = torch.zero_(model.b.data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    time = timeit.timeit(lambda : lyapunov_exponent(model, device), number=10)
    print(time)

    LEs = lyapunov_exponent(model, device)
    print(LEs)