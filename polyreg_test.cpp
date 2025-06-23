#include <vector>
#include <array>
#include <iostream>

extern "C" void* pwreg_f64d2_init(size_t numElements, double *data);
extern "C" void* pwreg_f64d3_init(size_t numElements, double *data);

extern "C" void pwreg_f64d2_delete(void *regression);
extern "C" void pwreg_f64d3_delete(void *regression);

extern "C" void pwreg_f64d2_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors);
extern "C" void pwreg_f64d3_reduce(void *regression, size_t numPieces, size_t *breakPoints, double *pieceModels, double *pieceErrors);

extern "C" void* pwreg_f64d2_copy(void *regression);
extern "C" void* pwreg_f64d3_copy(void *regression);

extern "C" void pwreg_f64d2_optimize(void *regression, double *data);
extern "C" void pwreg_f64d3_optimize(void *regression, double *data);

int testLinEx()
{
    std::vector<std::pair<size_t, std::array<double, 2>>> funcs
    {
        {10, { 1.0, 70.0}},
//        {11, {-2.0,  5.0}},
        {11, {-2.0, -250.0}},
        { 6, { 9.5, -7.0}},
        { 8, { 0.0,  0.0}},
        //{13, { 0.0, 30.0}},
        {13, { 3.0, 30000.0}},
        {20, {-1.0, -7.0}}
    };
    std::vector<std::array<double, 3>> data;
    for (size_t cur = 0; auto &func : funcs)
    {
        for (size_t x = cur; x < cur + func.first; x++)
        {
            data.push_back({1.0, double(x), double(x)*func.second[0] + double(func.second[1])});
        }
        cur += func.first;
    }
    auto reg = pwreg_f64d2_init(data.size(), data.data()->data());

    std::array<size_t, 6> bps;
    std::array<std::array<double, 2>, 6> models;
    std::array<double, 6> errors;

    pwreg_f64d2_reduce(reg, 6, bps.data(), models.data()->data(), errors.data());

    for (int i = 0; i < 6; i++)
    {
        std::cout << "Model " << i << ": (" << bps[i] << " - " << ((i+1 >= bps.size()) ? data.size() : bps[i+1]) << ")\n";
        std::cout << "  Params: ";
        for (auto val : models[i])
        {
            std::cout << val << " ";
        }
        std::cout << "\n" << "  Error: " << errors[i] << std::endl;
    }

    pwreg_f64d2_optimize(reg, data.data()->data());
    pwreg_f64d2_reduce(reg, 6, bps.data(), models.data()->data(), errors.data());

    std::cout << "Optimized segmentation:\n";
    for (int i = 0; i < 6; i++)
    {
        std::cout << "Model " << i << ": (" << bps[i] << " - " << ((i+1 >= bps.size()) ? data.size() : bps[i+1]) << ")\n";
        std::cout << "  Params: ";
        for (auto val : models[i])
        {
            std::cout << val << " ";
        }
        std::cout << "\n" << "  Error: " << errors[i] << std::endl;
    }

    pwreg_f64d2_delete(reg);
    return 0;
}

// test 2D
int test2d()
{
    std::vector<std::array<double, 3>> data;
    for (size_t i = 0; i < 21; i++)
    {
        double x = double(i)-10;
        data.push_back({ 1.0, x, (x >= 0.0) ? x : -x });
    }
    auto reg = pwreg_f64d2_init(data.size(), data.data()->data());

    std::array<size_t, 2> bps;
    std::array<std::array<double, 2>, 2> models;
    std::array<double, 2> errors;

    pwreg_f64d2_reduce(reg, 2, bps.data(), models.data()->data(), errors.data());

    for (int i = 0; i < 2; i++)
    {
        std::cout << "Model " << i << ": (" << bps[i] << " - " << ((i+1 >= bps.size()) ? data.size() : bps[i+1]) << ")\n";
        std::cout << "  Params: ";
        for (auto val : models[i])
        {
            std::cout << val << " ";
        }
        std::cout << "\n" << "  Error: " << errors[i] << std::endl;
    }

    pwreg_f64d2_delete(reg);
    return 0;
}

// test 3D
int test3d()
{
    auto constexpr Dimensions = 3;
    auto constexpr Segments = 2;
    std::vector<std::array<double, Dimensions+1>> data;
    for (size_t i = 0; i < 21; i++)
    {
        double x = double(i)-10;
        data.push_back({ 1.0, x, x*x, (x >= 0.0) ? x : -x });
    }
    auto reg = pwreg_f64d3_init(data.size(), data.data()->data());

    std::array<size_t, Segments> bps;
    std::array<std::array<double, Dimensions>, Segments> models;
    std::array<double, Segments> errors;

    pwreg_f64d3_reduce(reg, Segments, bps.data(), models.data()->data(), errors.data());

    for (int i = 0; i < Segments; i++)
    {
        std::cout << "Model " << i << ": (" << bps[i] << " - " << ((i+1 >= bps.size()) ? data.size() : bps[i+1]) << ")\n";
        std::cout << "  Params: ";
        for (auto val : models[i])
        {
            std::cout << val << " ";
        }
        std::cout << "\n" << "  Error: " << errors[i] << std::endl;
    }

    pwreg_f64d3_optimize(reg, data.data()->data());
    pwreg_f64d3_reduce(reg, Segments, bps.data(), models.data()->data(), errors.data());

    std::cout << "Optimized segmentation:\n";
    for (int i = 0; i < Segments; i++)
    {
        std::cout << "Model " << i << ": (" << bps[i] << " - " << ((i+1 >= bps.size()) ? data.size() : bps[i+1]) << ")\n";
        std::cout << "  Params: ";
        for (auto val : models[i])
        {
            std::cout << val << " ";
        }
        std::cout << "\n" << "  Error: " << errors[i] << std::endl;
    }

    pwreg_f64d3_delete(reg);
    return 0;
}

// test copy
int testCpy()
{
    std::array<double, 10*2> trash;
    std::vector<std::array<double, 3>> data;
    for (size_t i = 0; i < 21; i++)
    {
        double x = double(i)-10;
        data.push_back({ 1.0, x, (x >= 0.0) ? x : -x });
    }
    auto reg = pwreg_f64d2_init(data.size(), data.data()->data());

    auto reg2 = pwreg_f64d2_copy(reg);
    std::array<size_t, 2> bps;
    std::array<std::array<double, 2>, 2> models;
    std::array<double, 2> errors;
    pwreg_f64d2_reduce(reg2, 2, bps.data(), models.data()->data(), errors.data());

    std::array<size_t, 2> curBps{};
    std::array<std::array<double, 2>, 2> curModels{};
    std::array<double, 2> curErrors{};
    for (int i = 9; i > 2; i--)
    {
        pwreg_f64d2_delete(reg2);
        pwreg_f64d2_reduce(reg, i, (size_t*)trash.data(), trash.data(), trash.data());
        reg2 = pwreg_f64d2_copy(reg);
        pwreg_f64d2_reduce(reg2, 2, curBps.data(), curModels.data()->data(), curErrors.data());
        if (bps != curBps || models != curModels || errors != curErrors) throw;
        curErrors = {0,0};
        curModels = { std::array<double, 2>{0,0}, std::array<double, 2>{0,0} };
        curBps = {0,0};
    }
    pwreg_f64d2_delete(reg2);
    pwreg_f64d2_reduce(reg, 2, curBps.data(), curModels.data()->data(), curErrors.data());
    if (bps != curBps || models != curModels || errors != curErrors) throw;

    pwreg_f64d2_delete(reg);

    return 0;
}

void testlarge()
{
    auto constexpr Segments = 31;

    double y[] = {0.9174297454433946, 0.9058342304737526, 0.9011764553118977, 0.9022935638845552, 0.8936633971538561, 0.8853843873740076, 0.8707953106678076, 0.8721298627349026, 0.865693158877202, 0.853338149955058, 0.8441622223587681, 0.8439280942273065, 0.8426286146365672, 0.8259176321772975, 0.8304901765932087, 0.8216212797566822, 0.805871949365575, 0.8101700030161528, 0.7939783320702055, 0.7794776709058763, 0.7849432199399937, 0.7792596590967273, 0.7655938746706485, 0.7552351114344128, 0.7553090257293619, 0.7355868956643845, 0.727726570146196, 0.7348142783176653, 0.7153103817204974, 0.7091965098793493, 0.7169196030870743, 0.6919735877839334, 0.695348378548377, 0.6956178869702457, 0.6897771581167012, 0.6714657315388532, 0.6673170185133294, 0.6625231584164747, 0.6472620257970728, 0.6422040830506514, 0.6462001239207334, 0.6385393872722732, 0.6217043250421296, 0.6164814775005765, 0.6123126546394468, 0.6007522751871628, 0.6039697637270808, 0.59464351053592, 0.5870133503268372, 0.5752425955055168, 0.5748863823908976, 0.5573132967691289, 0.5622800818524991, 0.5436730618932496, 0.5415635881065529, 0.5348557817615835, 0.5286966416687164, 0.5257401795608068, 0.7728559681219795, 0.7338993605961883, 0.7131313754791273, 0.6887231660921255, 0.6637133278091162, 0.6395975684597401, 0.6215689740989114, 0.5995237068508826, 0.5675469686179431, 0.5427788651915963, 0.5216736829449093, 0.49064211001537106, 0.4663332136924282, 0.44352642115187196, 0.41740837806360903, 0.39080119320731493, 0.3670412750003359, 0.33756507843099104, 0.31222061596612594, 0.28896765238887284, 0.24872162561041417, 0.23972651226174985, 0.2420382005584683, 0.23622393511173778, 0.2319371686082564, 0.22813717664019234, 0.2273749794036474, 0.21962897225038747, 0.21212248470316042, 0.21402431075808193, 0.20357809580280578, 0.19688709502568916, 0.19288924974556715, 0.188597037972465, 0.18615366339236358, 0.179776513763373, 0.16695908811548227, 0.17469518364460776, 0.16267366884722584, 0.16082706750864714, 0.15140359074480422, 0.14530000926814307, 0.15216809807884948, 0.1390898667219658, 0.13818730056685036, 0.13298896845198485, 0.12636935075019773, 0.12036978772861312, 0.11113577177734293, 0.1077045407024276, 0.10185286226951312, 0.10282238674348489, 0.09697249812534431, 0.0916858174241221, 0.0831513249051891, 0.08344671725569307, 0.19781422208136973, 0.209959380538846, 0.2162317923164892, 0.23400519021476246, 0.24398725500465984, 0.2507495158394657, 0.2656865716684956, 0.28830671908044775, 0.2900489829672514, 0.3054946034483482, 0.32202851148029643, 0.3287041830765796, 0.339664230374864, 0.36221605441008164, 0.3806099945807943, 0.3925781689122928, 0.39620404614522253, 0.41358576279690024, 0.4254979256994174, 0.4329704293048764, 0.4521885782103476, 0.4580762453589021, 0.48035728301847697, 0.48731765717326325, 0.5005858779841119, 0.5042649082076239, 0.5312923763382907, 0.5419975326704694, 0.5477017867817408, 0.5656341674697453, 0.5764085476170092, 0.5922556786898651, 0.5985419043533423, 0.61565657154278, 0.6185301528623863, 0.6394207487841562, 0.652537071104854, 0.6614436042068885, 0.6774383463523557, 0.6969212279980388, 0.7032157906693313, 0.8902680411086263, 0.8777913728764456, 0.8562259633760503, 0.8456684937553797, 0.8261574371377568, 0.8103408096394418, 0.7950867264880402, 0.7819836128381249, 0.7661828904507023, 0.7450368024043241, 0.73452374573434, 0.722596564789126, 0.707188354037452, 0.6852509904257155, 0.6815806001063753, 0.6694421491186414, 0.6361242290126587, 0.6309195194060795, 0.6162801216512762, 0.601487206719367, 0.587355395521347, 0.5689696460707898, 0.5538750865048253, 0.5405671173400699, 0.5274232173575215, 0.5037261675450697, 0.496220263681929, 0.47977023981307365, 0.4718094055553958, 0.4460282183790987, 0.43064465775612726, 0.4255952137737146, 0.3887029534871753, 0.38748797428045323, 0.36776936082132106, 0.3537971152051799, 0.3316603925366553, 0.31778010897047154, 0.31248691900091813, 0.2857692009830135, 0.2770588916714112, 0.25713353095762737, 0.24662528894125704, 0.2263018740051238, 0.2186040426196957, 0.20591584480191272, 0.19571159298917187, 0.8509974716147679, 0.8379091037938188, 0.8189391130399121, 0.8060416739987606, 0.8015024557260694, 0.7914805909246667, 0.7788430256681532, 0.7587150086247297, 0.7561355741953618, 0.7487975538752207, 0.7155391679973425, 0.7169958975597631, 0.7012433127065896, 0.6845804904806002, 0.680389642150172, 0.6631730401283386, 0.6489608724018853, 0.6373496398368409, 0.6271714552291444, 0.6143023931479432, 0.6101707498121525, 0.596323530545326, 0.5820372306436357, 0.567070396061359, 0.5581498295517948, 0.5511170637329701, 0.5282840737869697, 0.5224470802632828, 0.4935567510398503, 0.5003477096867893, 0.48971270327394273, 0.4740378432818405, 0.45614116299080243, 0.4494859914388553, 0.43730000912409633, 0.42915179995528663, 0.41025369637172854, 0.40005717686462494, 0.37981252528572573, 0.37875789689721007, 0.3641315394196192, 0.34658232087092133, 0.3509138586302812, 0.3282857604654983, 0.32216622157190605, 0.2921132512728978, 0.2871860184986376, 0.2805108832033002, 0.27246863039266306, 0.2547066328099642, 0.25380899266569734, 0.23173995557927904, 0.21206382344211724, 0.20031160536945214};
    std::vector<std::array<double, 3>> data;
    data.reserve(sizeof(y)/sizeof(y[0]));
    for (size_t i = 0; i < sizeof(y)/sizeof(y[0]); i++)
    {
        data.push_back({ 1.0, double(i), y[i]});
    }
    auto reg = pwreg_f64d2_init(data.size(), data.data()->data());


    std::array<size_t, Segments> bps;
    std::array<std::array<double, 2>, Segments> models;
    std::array<double, Segments> errors;
    pwreg_f64d2_reduce(reg, Segments, bps.data(), models.data()->data(), errors.data());
    pwreg_f64d2_reduce(reg, 6, bps.data(), models.data()->data(), errors.data());

    for (int i = 0; i < Segments; i++)
    {
        std::cout << "Model " << i << ": (" << bps[i] << " - " << ((i+1 >= bps.size()) ? data.size() : bps[i+1]) << ")\n";
        std::cout << "  Params: ";
        for (auto val : models[i])
        {
            std::cout << val << " ";
        }
        std::cout << "\n" << "  Error: " << errors[i] << std::endl;
    }

    pwreg_f64d2_delete(reg);
}

int main()
{
    std::cout << "Testing 2d..." << std::endl;
    test2d();
    std::cout << "Testing 3d..." << std::endl;
    test3d();
    std::cout << "Testing linear example..." << std::endl;
    testLinEx();
    std::cout << "Testing large example..." << std::endl;
    testlarge();
    std::cout << "Testing cpy..." << std::endl;
    testCpy();
    std::cout << "Tip top!" << std::endl;
    return 0;
}
