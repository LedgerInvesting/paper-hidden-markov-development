functions {
  int isin(int i, array[] int x) {
    for (n in x) {
      if (n == i) {
        return 1;
      }
    }
    return 0;
  }
}
