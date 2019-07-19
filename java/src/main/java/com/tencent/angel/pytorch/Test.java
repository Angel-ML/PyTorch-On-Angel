package com.tencent.angel.pytorch;

import com.tencent.angel.pytorch.torch.TorchModel;

public class Test {
  public static void main(String[] argv) {
    System.loadLibrary("torch_angel");
    TorchModel.setPath("gcn.pt");
    System.out.println(TorchModel.get().getParametersTotalSize());
  }

}
