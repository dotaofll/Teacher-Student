#include <vector>
#include <tuple>

using namespace std;

class solution
{
public:
    auto selectWithWeigh(vector<int> &weight, vector<int> &sequence)
    {
        vector<int> res;
        size_t N = weight.size();
        for (auto &seq : sequence)
        {
            if (N%2 == 0){
                if(seq == (N/2)){
                    auto leftSide=this->_CountWeight(weight,seq,true);
                    auto rightSide = this->_CountWeight(weight,seq,false);
                    if(get<1>(leftSide) >= get<1>(rightSide)){
                        res.push_back(get<1>(leftSide)+weight[seq]);
                    }
                    res.push_back(get<1>(rightSide)+weight[seq]);
                }else{
                    if(seq < (N/2)){
                        auto [_count,_sum] = this->_CountWeight(weight,seq,false);
                        res.push_back(_sum+weight[seq]);
                    }
                    auto [_count,_sum] = this->_CountWeight(weight,seq,true);
                    res.push_back(_sum+weight[seq]);
                }
            }else{
                if(seq < (N/2)){
                    auto [_count,_sum] = this->_CountWeight(weight,seq,false);
                    res.push_back(_sum+weight[seq]);
                }
                auto [_count,_sum] = this->_CountWeight(weight,seq,true);
                res.push_back(_sum+weight[seq]);
            }
        }
        return res;
    }

private:
    auto _CountWeight(vector<int> &weight, int pos, bool left = true)
    {
        auto _count{0};
        auto _sum{0};

        if (left == true)
        {
            for (int i = 0; i < pos - 1; i++)
            {
                _sum += weight[i];
                _count++;
            }
        }
        else
        {
            for (int i = pos + 1; i < weight.size(); i++)
            {
                _sum += weight[i];
                _count++;
            }
        }
        return make_tuple(_count, _sum);
    }
}