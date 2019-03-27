while true;
do
  ./nodecore_pow_cuda -o pool.veriblockpool.com:8501 -u V8wB8DP56Cs9VrwyrXVsLTQ3e5Ngj5 -d 0 -bs 512 -tbs 1024;
  echo "Restarting PoW miner!";
  sleep 5;
done;
