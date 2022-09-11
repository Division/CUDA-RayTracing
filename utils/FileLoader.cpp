namespace Loader 
{
	std::optional<std::vector<uint8_t>> LoadFile(const std::wstring& filename) 
	{
		std::ifstream file(filename, std::ios::binary | std::ios::ate);
		if (!file.good())
			return std::nullopt;

		std::streamsize size = file.tellg();
		file.seekg(0, std::ios::beg);

		auto result = std::vector<uint8_t>(size);

		if (file.read((char*)result.data(), size)) {
			return result;
		}
		else {
			return std::nullopt;
		}
	}

}