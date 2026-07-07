// Unit tests for the CRC, framing, and incremental decoder.

#include <gtest/gtest.h>

#include "surgical_hand_serial/crc16.hpp"
#include "surgical_hand_serial/packet.hpp"

namespace shs = surgical_hand_serial;

TEST(Crc16, KnownVector) {
  // CRC16-CCITT-FALSE("123456789") = 0x29B1.
  const uint8_t data[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  EXPECT_EQ(shs::crc16Ccitt(data, sizeof(data)), 0x29B1);
}

TEST(Packet, EncodeLayout) {
  shs::Packet packet;
  packet.seq = 7;
  packet.command = static_cast<uint8_t>(shs::CommandId::kPing);
  const auto frame = shs::encodePacket(packet);

  ASSERT_EQ(frame.size(), 7u);  // sync(2) + len + seq + cmd + crc(2)
  EXPECT_EQ(frame[0], shs::kSync0);
  EXPECT_EQ(frame[1], shs::kSync1);
  EXPECT_EQ(frame[2], 2);  // len = seq + cmd, no payload
  EXPECT_EQ(frame[3], 7);
  EXPECT_EQ(frame[4], 0x01);
}

TEST(Packet, RoundTripByteAtATime) {
  shs::Packet packet;
  packet.seq = 42;
  packet.command = static_cast<uint8_t>(shs::CommandId::kReadMotorState);
  packet.payload = {1, 2, 3, 4, 5};
  const auto frame = shs::encodePacket(packet);

  shs::PacketDecoder decoder;
  std::optional<shs::Packet> decoded;
  for (size_t i = 0; i < frame.size(); ++i) {
    decoded = decoder.feed(frame[i]);
    if (i + 1 < frame.size()) {
      EXPECT_FALSE(decoded.has_value()) << "packet completed early at byte " << i;
    }
  }
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->seq, 42);
  EXPECT_EQ(decoded->command, static_cast<uint8_t>(shs::CommandId::kReadMotorState));
  EXPECT_EQ(decoded->payload, (std::vector<uint8_t>{1, 2, 3, 4, 5}));
  EXPECT_EQ(decoder.discardedBytes(), 0u);
}

TEST(Packet, GarbagePrefixIsDiscarded) {
  shs::Packet packet;
  packet.seq = 1;
  packet.command = static_cast<uint8_t>(shs::CommandId::kHeartbeat);
  auto frame = shs::encodePacket(packet);

  std::vector<uint8_t> stream = {0x00, 0xFF, 0x13, 0x37};
  stream.insert(stream.end(), frame.begin(), frame.end());

  shs::PacketDecoder decoder;
  std::optional<shs::Packet> decoded;
  for (const uint8_t byte : stream) {
    if (auto p = decoder.feed(byte)) {
      decoded = p;
    }
  }
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->command, static_cast<uint8_t>(shs::CommandId::kHeartbeat));
  EXPECT_EQ(decoder.discardedBytes(), 4u);
}

TEST(Packet, CorruptedCrcIsRejectedThenRecovers) {
  shs::Packet packet;
  packet.seq = 9;
  packet.command = static_cast<uint8_t>(shs::CommandId::kPing);
  auto bad_frame = shs::encodePacket(packet);
  bad_frame.back() ^= 0xFF;  // corrupt CRC high byte
  const auto good_frame = shs::encodePacket(packet);

  shs::PacketDecoder decoder;
  std::optional<shs::Packet> decoded;
  for (const uint8_t byte : bad_frame) {
    decoded = decoder.feed(byte);
    EXPECT_FALSE(decoded.has_value());
  }
  EXPECT_GT(decoder.discardedBytes(), 0u);

  for (const uint8_t byte : good_frame) {
    if (auto p = decoder.feed(byte)) {
      decoded = p;
    }
  }
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->seq, 9);
}

TEST(Packet, TwoPacketsBackToBack) {
  shs::Packet first;
  first.seq = 1;
  first.command = static_cast<uint8_t>(shs::CommandId::kPing);
  shs::Packet second;
  second.seq = 2;
  second.command = static_cast<uint8_t>(shs::CommandId::kGetStatus);
  second.payload = {0xAB};

  auto stream = shs::encodePacket(first);
  const auto second_frame = shs::encodePacket(second);
  stream.insert(stream.end(), second_frame.begin(), second_frame.end());

  shs::PacketDecoder decoder;
  std::vector<shs::Packet> decoded;
  for (const uint8_t byte : stream) {
    if (auto p = decoder.feed(byte)) {
      decoded.push_back(*p);
    }
  }
  ASSERT_EQ(decoded.size(), 2u);
  EXPECT_EQ(decoded[0].seq, 1);
  EXPECT_EQ(decoded[1].seq, 2);
  EXPECT_EQ(decoded[1].payload, std::vector<uint8_t>{0xAB});
}

TEST(Packet, MaxPayloadOkOversizeThrows) {
  shs::Packet packet;
  packet.seq = 3;
  packet.command = static_cast<uint8_t>(shs::CommandId::kSetGoalPosition);
  packet.payload.assign(shs::kMaxPayloadSize, 0x55);
  const auto frame = shs::encodePacket(packet);  // must not throw

  shs::PacketDecoder decoder;
  std::optional<shs::Packet> decoded;
  for (const uint8_t byte : frame) {
    if (auto p = decoder.feed(byte)) {
      decoded = p;
    }
  }
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->payload.size(), static_cast<size_t>(shs::kMaxPayloadSize));

  packet.payload.push_back(0x55);
  EXPECT_THROW(shs::encodePacket(packet), std::length_error);
}

TEST(Packet, LittleEndianHelpers) {
  std::vector<uint8_t> buffer;
  shs::appendU16(buffer, 0x1234);
  shs::appendU32(buffer, 0xDEADBEEF);
  shs::appendI16(buffer, -2);
  shs::appendI32(buffer, -100000);

  EXPECT_EQ(buffer[0], 0x34);  // little-endian
  EXPECT_EQ(buffer[1], 0x12);
  EXPECT_EQ(shs::readU16(buffer, 0), 0x1234);
  EXPECT_EQ(shs::readU32(buffer, 2), 0xDEADBEEF);
  EXPECT_EQ(shs::readI16(buffer, 6), -2);
  EXPECT_EQ(shs::readI32(buffer, 8), -100000);
  EXPECT_THROW(shs::readU32(buffer, 9), std::out_of_range);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
